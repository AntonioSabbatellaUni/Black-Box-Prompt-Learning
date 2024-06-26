import numpy as np
import time
import torch
# (Ant) Import the function that parse the arguments
import argparse
from transformers import AutoTokenizer
from datasets import load_metric
import logging
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from transformers.models.roberta.configuration_roberta import RobertaConfig

from transformers import RobertaForMaskedLM

from botorch.test_functions.base import BaseTestProblem
import os

import datasets
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
import random
from torch.utils.data import DataLoader

from botorch.models import SingleTaskGP#, FixedNoiseGP, ModelListGP
# from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
# ExactMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
#GaussianProcessRegressor
#acquisition_function: UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement
import warnings






# (Ant) List of the dataset that are domain-specific - nedeed in self.config = RobertaConfig ... 
LABEL2ID_CONFIG = {
    "mnli": {" no": 0, " maybe": 1, " yes": 2},
    "qqp": {" no": 0, " yes": 1},
    "sst2": {" terrible": 0, " great": 1},
    "mrpc": {" no": 0, " yes": 1},
    "cola": {" no": 0, " yes": 1},
    "wnli": {" no": 0, " yes": 1},
    "qnli": {" yes": 0, " no": 1},
    "rte": {" yes": 0, " no": 1},
    "CI": {' background': 0, ' comparison': 1, ' extension': 2, ' future': 3, ' motivation': 4, ' use': 5},
    "SE": {' comparison': 0, ' conjunction': 1, ' evaluation': 2, ' feature': 3, ' hyponym': 4, ' part': 5, ' function': 6},
    "RCT": {' background': 0, ' conclusion': 1, ' method': 2, ' objective': 3, ' result': 4} ,
    "HP": {' unhelpful': 0, ' helpful': 1}, # review helpfulness
    "imdb": {" terrible": 0, " great": 1},
    "cr": {" terrible": 0, " great": 1},
    "mr": {" terrible": 0, " great": 1},
    "mpqa": {" terrible": 0, " great": 1}
}
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TEMPLATE_CONFIG = {
    "mnli": " entailment? [MASK].",
    "qqp": "? [MASK],",
    "sst2": " It was [MASK].",
    "mrpc": "? [MASK],",
    "cola": " correct? [MASK].",
    "wnli": " entailment? [MASK].",
    "qnli": " entailment? [MASK].",
    "rte": " entailment? [MASK].",
    "CI": " What is the intent? [MASK].", 
    "SE": " What is the relation? [MASK].",
    "RCT": " It is [MASK]. ",
    "HP": " It is [MASK].",
    "imdb": "It was [MASK].",
    "cr": "It was [MASK].",
}

LABEL_CONVERT = {
    "mnli": {0: ' no', 1: ' maybe', 2: ' yes'},
    "qqp": {0: ' no', 1: ' yes'},
    "sst2": {0: ' terrible', 1: ' great'},
    'mrpc': {0: ' no', 1: ' yes'},
    'cola': {0: ' no', 1: ' yes'},
    'wnli': {0:  ' no', 1: ' yes'},
    'qnli': {0: ' yes', 1: ' no'},
    'rte': {0: ' yes', 1: ' no'},
    'CI': {'Background': ' background', 'CompareOrContrast': ' comparison', 'Extends': ' extension', 'Future': ' future', 'Motivation': ' motivation', 'Uses': ' use'},
    'SE': {'COMPARE': ' comparison', 'CONJUNCTION': ' conjunction', 'EVALUATE-FOR': ' evaluation', 'FEATURE-OF': ' feature', 'HYPONYM-OF': ' hyponym', 'PART-OF': ' part', 'USED-FOR': ' function'},
    'RCT': {'BACKGROUND': ' background', 'CONCLUSIONS': ' conclusion', 'METHODS': ' method', 'OBJECTIVE': ' objective', 'RESULTS': ' result'},
    'HP': {False: ' unhelpful', True: ' helpful'},
}
DOMAIN_DATASET = ['CI', 'SE', 'RCT', 'HP']

# (Ant) Function called by pmi() to parse the arguments
def parse_args():
    all_args_list_default = {'task_name': None, 'file_name': None, 'low_resource': False, 'ce_loss': True, 'sample_size': 20, 'prompt_length': 6, 'prompt_learning_rate': 5e-5, 'prompt_search_space': 20, 'num_train_epochs': 30, 'ckpt_path': './ckpts', 'margin': 1, 'trial': False, 'use_wandb': True, 'cuda': 0, 'max_length': 450, 'pad_to_max_length': False, 'per_device_train_batch_size': 128, 'per_device_eval_batch_size': 32, 'model_name_or_path': 'roberta-large', 'use_slow_tokenizer': False, 'weight_decay': 0.1, 'max_train_steps': None, 'gradient_accumulation_steps': 1, 'lr_scheduler_type': 'linear', 'num_warmup_steps': 100, 'output_dir': None, 'seed': 42, 'k_shot': -1, 'use_ngram': True, 'api_limit': 8000}

    args_selected = {"task_name": "mrpc", "per_device_train_batch_size": 128, "per_device_eval_batch_size": 16, "weight_decay": 0.1, "seed": 42, "k_shot": 16, "prompt_learning_rate": 1e-4, "sample_size": 20, "prompt_length": 10, "prompt_search_space": 200, "api_limit": 8000, "ce_loss": True}

    args ={}
    for arg in all_args_list_default.keys():
        if arg not in args_selected.keys():
            args[arg] = all_args_list_default[arg]
        else:
            args[arg] = args_selected[arg]

    # args to object
    args = argparse.Namespace(**args)

    return args

# (Ant) Funcrion called by the function evaluate, taken from run_glue_discrete_LM.py
def pmi():
    args = parse_args()
    result=[]
    if args.file_name:
        with open("./pmi/" + args.file_name.lower() + ".txt",'r') as f:
            for line in f:
                result = result + (list(line.strip('\n').split(',')))
    elif args.task_name:
        path =  "/workspaces/basic-python/Black-Box-Prompt-Learning/"
        with open(path + "/pmi/" + args.task_name.lower() + ".txt",'r') as f:
            
            for line in f:
                result = result + (list(line.strip('\n').split(',')))

    # (Ant) This is the original code, but it doesn't work because the path is wrong
    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(map(int, unique))
    return ngram_index_list

# (Ant) Function called by the function evaluate
results = []


class BoPrompter(BaseTestProblem):

    def __init__(self):
        self.args = parse_args()
        args = self.args 
        ngram_list = pmi()

        # data loader ( form 256 to 500 of original code )
        assert args.task_name != 'stsb'
        ce_loss_string = 'True' if args.ce_loss else 'False'

        # specify a unique experiment_id for load_metric() otherwise will cause ERROR when having multiple run on a same server!
        task_name = args.task_name if args.task_name else args.train_file
        args.unique_task_name = task_name.replace("/", ".")
        args.experiment_id = task_name + str(args.prompt_length) + str(args.prompt_learning_rate) \
                            + str(args.num_train_epochs) + str(args.seed) + str(args.prompt_search_space) + ce_loss_string #'dataset/CI/train.csv1020.0013042160.01falseFALSE'

        if args.use_wandb and False: # not needed for now
            args.group_name = "RoBERTa_BDPL_" + task_name
            wandb.init(config=args, project="blackbox_prompt", group=args.group_name)

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator()
        
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # download the dataset.
        if args.task_name is not None:
            if args.task_name in task_to_keys.keys():
                raw_datasets = load_dataset("glue", args.task_name)
            else:
                raise(NotImplementedError)
        else:
            # Loading the dataset from local csv or json file.
            data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            if args.test_file is not None:
                data_files["test"] = args.test_file
            extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files)

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if args.task_name:
            label_to_id = LABEL2ID_CONFIG[args.task_name] # -> debugging label_to_id:{' no': 0, ' yes': 1}
        elif args.file_name:
            label_to_id = LABEL2ID_CONFIG[args.file_name]

        num_labels = len(label_to_id)

        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

        # init model
        model = RobertaForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        
        # args.device = torch.device("cuda", args.cuda)
        args.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        model.to(args.device)

        if label_to_id is not None:
            model.config.label2id = label_to_id # -> debugging model.config.label2id:{' no': 0, ' yes': 1}
            model.config.id2label = {id: label for label, id in config.label2id.items()} # -> debugging model.config.id2label:{0: ' no', 1: ' yes'}

        # @counter
        def train_api_request(input_ids=None, attention_mask=None):
            sequence_output = model(input_ids=input_ids, attention_mask=attention_mask)
            return sequence_output

        prompt_length = args.prompt_length
        # hingeloss = MarginLoss(margin=args.margin, target=False)
        ce_loss = CrossEntropyLoss()

        # Preprocessing the datasets
        if args.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[args.task_name] # -> debugging sentence1_key:'sentence1', sentence2_key:'sentence2'
        else:
            non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        padding = "max_length" if args.pad_to_max_length else False # -> debugging padding:False

        def preprocess_function(examples):
            # Tokenize the texts
            if args.low_resource:
                train_random_samples = random.sample(range(0, len(examples["label"])), len(examples["label"])//10)
                for key in examples.keys():
                    examples[key] = [examples[key][k] for k in train_random_samples]

            if args.file_name == 'HP':
                for k in range(len(examples["text_a"])):
                    if examples["text_a"][k] == None:
                        examples["text_a"].remove(examples["text_a"][k])
                        examples["label"].remove(examples["label"][k])
                        break

            if args.task_name is not None:
                template_cfg = TEMPLATE_CONFIG[args.task_name]
            elif args.file_name is not None:
                template_cfg = TEMPLATE_CONFIG[args.file_name]
            template_base = template_cfg.replace('[MASK]', tokenizer.mask_token)

            if sentence2_key:
                sent1_list = []
                for sent1 in examples[sentence1_key]:
                    sent1_list.append(sent1 + template_base)
                texts = (sent1_list, examples[sentence2_key])
            else:
                template = [template_base] * len(examples[sentence1_key])
                texts = (examples[sentence1_key], template)
            result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True, add_special_tokens=False)

            texts = []
            template = [template_base] * len(examples[sentence1_key])
            if sentence2_key:
                for tuple_ in list(zip(examples[sentence1_key], template, examples[sentence2_key])):
                    sent_1 = tokenizer.tokenize(tuple_[0])[:200]
                    new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                    sent_2 = tokenizer.tokenize(tuple_[2])[:200]
                    new_sent_2 = tokenizer.convert_tokens_to_string(sent_2)
                    texts.append(new_sent_1 + tokenizer.sep_token + new_sent_2 + tuple_[1])
            else:
                for tuple_ in list(zip(examples[sentence1_key], template)):
                    sent_1 = tokenizer.tokenize(tuple_[0])[:400]
                    new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                    texts.append(new_sent_1 + tuple_[1])
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)

            if args.task_name:
                label_list = []
                for raw_label in examples["label"]:
                    label = LABEL_CONVERT[args.task_name][raw_label]
                    target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                    label_list.append(target_encodings[0])
                result["labels"] = torch.tensor(label_list)

            elif args.file_name in DOMAIN_DATASET:
                label_list = []
                for raw_label in examples["label"]:
                    label = LABEL_CONVERT[args.file_name][raw_label]
                    target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                    label_list.append(target_encodings[0])
                result["labels"] = torch.tensor(label_list)
            else:
                target_encodings = tokenizer.batch_encode_plus(examples["label"], add_special_tokens=False)
                result["labels"]= torch.tensor(target_encodings['input_ids']).squeeze(dim=1).to(args.device)
                
            return result

        def preprocess_function_k_shot(examples):
            random_indices = list(range(0, len(examples["label"])))
            random.shuffle(random_indices)

            new_examples = {}
            for key in examples.keys():
                new_examples[key] = []
            label_count = {}

            for index in random_indices:
                label = examples['label'][index]
                if label not in label_count:
                    label_count[label] = 0

                if label_count[label] < args.k_shot:
                    for key in examples.keys():
                        new_examples[key].append(examples[key][index])
                    label_count[label] += 1
            
            print("Finish few-shot sampling!")

            result = preprocess_function(new_examples)
            return result

        with accelerator.main_process_first():
            if args.k_shot >= 0: # k-shot learning value : 16
                # k-shot learning
                raw_train_dataset_split = raw_datasets["train"].train_test_split(test_size=0.5) # raw_datasets = load_dataset("glue", args.task_name)
                raw_train_dataset = raw_train_dataset_split['train']
                raw_eval_dataset = raw_train_dataset_split['test']
                train_dataset = raw_train_dataset.map(
                    preprocess_function_k_shot,
                    batched=True,
                    batch_size=100000,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
                eval_dataset = raw_eval_dataset.map(
                    preprocess_function_k_shot,
                    batched=True,
                    batch_size=100000,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
                if args.task_name == 'mnli':
                    test_dataset = raw_datasets["validation_matched"].map(
                        preprocess_function,
                        batched=True,
                        remove_columns=raw_datasets["train"].column_names,
                        load_from_cache_file=False,
                        desc="Running tokenizer on dataset",
                    )
                    test_dataset_mm = raw_datasets["validation_mismatched"].map(
                        preprocess_function,
                        batched=True,
                        remove_columns=raw_datasets["train"].column_names,
                        load_from_cache_file=False,
                        desc="Running tokenizer on dataset",
                    )
                else:
                    test_dataset = raw_datasets["validation"].map(
                        preprocess_function,
                        batched=True,
                        remove_columns=raw_datasets["train"].column_names,
                        load_from_cache_file=False,
                        desc="Running tokenizer on dataset",
                    )
            else:
                train_dataset = raw_datasets["train"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
                eval_dataset = raw_datasets["validation"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
                test_dataset = raw_datasets["test" if args.file_name != None else "validation"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
            print("length of train data",len(train_dataset))
            print("length of eval data",len(eval_dataset))
            print("length of test data",len(test_dataset))
            
            global train_dataset_example # for debugging
            train_dataset_example = train_dataset

        # Log a few random samples from the training set:
        # for index in random.sample(range(len(train_dataset)), 3):
            # logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        if args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        if args.task_name == 'mnli':
            test_dataloader_mm = DataLoader(test_dataset_mm, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
            test_dataloader_mm = accelerator.prepare(test_dataloader_mm)
        else:
            test_dataloader_mm = None
        model, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader, test_dataloader)



        # ***** END OF DATA LOADING snippet *****

        if args.task_name is not None:
            metric = load_metric("glue", args.task_name) #, experiment_id=args.experiment_id)
        elif args.file_name in DOMAIN_DATASET:
            metric = load_metric('f1', args.experiment_id)
        else:
            metric = load_metric('accuracy', args.experiment_id)

        self.metric = load_metric("glue", args.task_name)
        self.logger = logging.getLogger(__name__)

        # needed in evaluate_true, not sure if this is the correct way to get epoch 
        self.epoch = 0
        self.ngram_list = ngram_list
        # self.

        
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator()

        # just to not throw error _bounds not defined ***
        # ten times [0, 3143]
        self._bounds = [[0, 3143]] * 10
        self.dim  = 10

        super().__init__()
        setattr(self, "model", model)
        setattr(self, "accelerator", accelerator)
        setattr(self, "ce_loss", CrossEntropyLoss())

        setattr(self, "train_dataloader", train_dataloader)
        setattr(self, "eval_dataloader", eval_dataloader)
        setattr(self, "test_dataloader", test_dataloader)

        setattr(self, "tokenizer", tokenizer)
        setattr(self, "config", config)



    def evaluate_true(self, X): # funzione cge valuta | X tensor "list of list" token di input  | a tensor list of scores for each element in X
        
        #(Ant) In this case X the output of the Algorithm, not the direct prediction ( need argmax )

        # Get predicted prompt indices by taking argmax of probabilities
        # prompts_discrete_indices = X.argmax(1) 
        prompts_discrete_indices = X 

        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        metric = self.metric

        epoch = self.epoch # NOT SURE IF THIS IS CORRECT WAY TO GET EPOCH, CHECK

        ngram_list = self.ngram_list

        # ce_loss = self.ce_loss
        ce_loss = self.ce_loss

        accelerator = self.accelerator

        prompt_length = args.prompt_length
        config = self.config

        # If using n-gram prompts, convert indices to n-gram sequences
        if args.use_ngram:
            prompts_discrete_indices_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()

            # dimensions = 81
            # indices_list = [indices_list[i:i + dimensions] for i in range(0, len(indices_list), dimensions)]
            # indices_list = indices_list[0]


            for idx in indices_list:#[0]: # *****hard coded for testing 1 sentence
                # idx = 5 if idx >= 4985 else idx # **
                prompts_discrete_indices_ngram_list.append(ngram_list[idx])
            prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

        # Iterate through batches
        for step, batch in enumerate(self.eval_dataloader):

            # Stop after 100 batches if trial run
            if args.trial and step >= 100:
                break   
            
            # Get batch size
            bsz = len(batch['input_ids']) # bsz = 16, torch.Size([16, 71])

            # Concatenate prompts and input ids
            if args.use_ngram:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1) 
            else:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)

            # Concatenate attention masks    
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            # Get mask token positions
            mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id)
            mask_pos = torch.tensor(mask_pos[-1])

            # Get label mappings 
            label_to_id = model.config.label2id

            print("levaluating: ", batch['input_ids'].shape)
            # Forward pass
            with torch.no_grad():
                sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
            torch.cuda.empty_cache()
            # Extract logits at mask positions
            last_hidden_state = sequence_output[0].squeeze() 
            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            # Get labels
            label = batch["labels"].to(args.device)

            # Map labels to ids
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]

            # Convert labels
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val

            # Filter logits  
            interest_index = list(label_map.keys())
            logits = logits[:, interest_index]

            # Compute loss
            eval_loss_c = ce_loss(logits.view(-1, config.num_labels), converted_target)

            print("Cross Entropy Loss: ", eval_loss_c)
            return eval_loss_c # return the loss 

            # Get predictions
            predictions = logits.argmax(dim=-1)

            
            # Update metrics
            metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(converted_target),
            )

            # Compute overall metrics
            if args.file_name in DOMAIN_DATASET:
                eval_metric = metric.compute(average='macro')
            else:
                eval_metric = metric.compute()

            logger = self.logger
            # Log results
            logger.info("** eval **")
            logger.info(f"epoch {epoch}: {eval_metric}")

            # Extract key metric  
            if args.task_name == 'cola':
                key = 'matthews_correlation'
            elif args.task_name in ['mnli', 'sst2', 'wnli', 'rte', 'qnli'] or args.file_name in ['MR', 'CR']:
                key = 'accuracy'
            else:
                key = 'f1'
            eval_result = eval_metric[key]

            # Append to results
            results.append(eval_result)

            return eval_result
    
    def generate_initial_data(self, n=10):
        # random_seed = 0
        random_seed = 42 + 2
        print("Random seed: ", random_seed)
        torch.manual_seed(random_seed)

        # Expected all inputs to share the same dtype, the train_x so is transformed in a float tensor
        train_x = torch.randint(0, 3143, (n, 10), dtype=torch.float32) # ***tensor of tensor tensor([[2335 ...x10... 810], [...], ... ,[...]])

        train_y = [ [self.evaluate_true(torch.tensor(x))] for x in train_x ]
        train_y = torch.tensor(train_y) # ***tensor of tensor tensor([[0.0000], [0.0000], ... , [0.0000]])


        return train_x, train_y

    def init_model(self ,train_x, train_y, state_dict=None):
        # gp = GaussianProcessRegressor(train_x, train_y)
        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if state_dict is not None:
            gp.load_state_dict(state_dict)
        return gp, mll

    def optimize_acquisition_function(self, acquisition_function, bounds, gp, num_restarts=10, raw_samples=100):
        if(acquisition_function == "ucb"):
            ucb = UpperConfidenceBound(gp, beta=0.4, maximize=False) # maximize=True for accuracy
        elif(True): # for now use always ucb
            ucb = UpperConfidenceBound(gp, beta=0.4, maximize=False) 
        candidate, _ = optimize_acqf(ucb, bounds=bounds, q=1, num_restarts=20, raw_samples=50)
        new_point = candidate.detach()#pu().numpy()
        return torch.round(new_point).int() # dafault is 0
    
    def train_loop(self, verbose = True, npoint= 2):
        bounds = torch.tensor([[0] * 10, [3143] * 10], dtype=torch.float32)
        if verbose:
            print("*** Training loop ***")

        train_x, train_y = self.generate_initial_data()
        print("Initial train_x: ", train_x)
        print("Initial train_y: ", train_y)
        gp, mll = self.init_model(train_x, train_y)
        loss_value_list = []
        for j in range(npoint):
            if verbose:
                print(f"* Evaluating point n {j}")
            acquisition_function="ucb"
            new_point = self.optimize_acquisition_function(acquisition_function, bounds, gp)
            # new_point = new_point.squeeze() # from 2d [[11, 23, 32...]] to 1d [11, 23, 32...]
            current_loss = self.evaluate_true(new_point.squeeze())

            if verbose:
                print(f"* New point: {new_point}")
            train_x = torch.cat((train_x, new_point), 0)
            train_y = torch.cat((train_y, current_loss.unsqueeze(0).unsqueeze(0)), 0)
            gp, mll = self.init_model(train_x, train_y, gp.state_dict())
            loss_value_list.append({'point': new_point, 'loss': current_loss})
        return gp, mll, train_x, train_y# ,loss_value_list

if __name__ == "__main__":
    test = BoPrompter()
    print("testing the BoPrompter")
    # start = time.time()
    # # tensor = torch.tensor([194, 122, 122,  92, 108,  71,  63, 151,  21,  49])
    # tensor = torch.tensor([0, 1, 2,  3000, 108,  71,  63, 151,  21,  49])

    # # res = test(tensor)
    # # print(res)
    # print("time taken: ", time.time() - start)


    # Ignora tutti i warning
    warnings.filterwarnings("ignore")


    gp, mll, train_x, train_y = test.train_loop(verbose=True, npoint=100)
    print("train_x: ", train_x)
    print("train_y: ", train_y)

    # test = BoPrompter()
    # # Get first sample 
    # train_dataloader = test.train_dataloader
    # for batch in train_dataloader:
    #     sample = batch['input_ids'][0]
    #     break
    # sample = sample.view(1, -1) 
    # test(sample)


#dim 5 50 iter 100
# grafico minimo