from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import time
from sympy import Tuple
import torch
import argparse
from transformers import AutoTokenizer
from datasets import load_metric
import logging
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from transformers.models.roberta.configuration_roberta import RobertaConfig

from transformers import RobertaForMaskedLM

from base import SingleObjectiveProblem
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
from bbopy.problems.base import SingleObjectiveProblem

from datasets_utils import create_dataset

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
import warnings
from tqdm import tqdm
from tqdm import tqdm
import os
import time
from llm_api_call import chat_LLM
import pickle
import datetime

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
    "mpqa": {" terrible": 0, " great": 1},
    "sentiment": {" negative": 0, " positive": 1},
    "sentence_similarity": {"0 - definitely not": 0, "1 - probably not": 1, "2 - possibly": 2, "3 - probably": 3, "4 - almost perfectly": 4, "5 - perfectly": 5},
    "word_in_context": {"not the same": 0, "same": 1}#, "no": 0, "yes": 1, "true": 1, "false": 0}
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
    "sentiment": ("sentence", None),
    "sentence_similarity": ("sentence", None),
    "word_in_context": ("sentence", None),
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
    "sentiment": "It was [MASK].",
    "sentence_similarity": "Similarity score: [MASK].",
    "word_in_context": "Meaning is: [MASK].",
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
    'sentiment': {0: ' negative', 1: ' positive'},
    'sentence_similarity': {0: '0 - definitely not', 1: '1 - probably not', 2: '2 - possibly', 3: '3 - probably', 4: '4 - almost perfectly', 5: '5 - perfectly'},
    'word_in_context': {0: 'not the same', 1: 'same'}
}
DOMAIN_DATASET = ['CI', 'SE', 'RCT', 'HP']
EXTERNAL_DATASET = ['sentiment', 'sentence_similarity', 'word_in_context']
results = []
global llm_logger
llm_logger = {}

def parse_args(args_selected: Optional[dict] = None) -> argparse.Namespace:
    """
    This function parses the selected arguments and returns an argparse.Namespace object.

    Args:
        args_selected (dict): A dictionary of selected arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    all_args_list_default = {'task_name': 'sentiment', 'file_name': None, 'low_resource': False, 'ce_loss': True, 'sample_size': 20, 'prompt_length': 10, 'prompt_learning_rate': 5e-5, 'prompt_search_space': 20, 'num_train_epochs': 30, 'ckpt_path': './ckpts', 'margin': 1, 'trial': False, 'use_wandb': True, 'cuda': 0, 'max_length': 450, 'pad_to_max_length': False, 'per_device_train_batch_size': 128, 'per_device_eval_batch_size': 32, 'model_name_or_path': 'roberta-large', 'use_slow_tokenizer': False, 'weight_decay': 0.1, 'max_train_steps': None, 'gradient_accumulation_steps': 1, 'lr_scheduler_type': 'linear', 'num_warmup_steps': 100, 'output_dir': None, 'seed': 42, 'k_shot': -1, 'use_ngram': True, 'api_limit': 8000}
    if args_selected is None:
        args_selected = {}
    
    args ={}
    for arg in all_args_list_default.keys():
        if arg not in args_selected.keys():
            args[arg] = all_args_list_default[arg]
        else:
            args[arg] = args_selected[arg]

    # args to object
    args = argparse.Namespace(**args)

    return args

def pmi(args_selected=None) -> list:
    """
    This function reads a file and returns a list of ngram index values.

    Args:
        args_selected (list): A list of selected arguments.

    Returns:
        list: A list of ngram index values.
    """
    # /.../pmi/pmi_cola_gpt.txt | readable n-gram 
    args = parse_args(args_selected)
    result = []
    if args.file_name:
        with open(os.path.join(os.path.dirname(__file__), "pmi", f"pmi_{args.file_name.lower()}_gpt.txt"), 'r') as f:
            for line in tqdm(f, desc=f"Reading file pmi_{args.file_name.lower()}_gpt.txt"):
                result = result + (list(line.strip('\n').split(',')))
    elif args.task_name:
        path = os.path.join(os.path.dirname(__file__), "pmi")
        with open(os.path.join(path, f"pmi_{args.task_name.lower()}_gpt.txt"), 'r') as f:
            for line in tqdm(f, desc=f"Reading file pmi_{args.task_name.lower()}_gpt.txt"):
                result = result + (list(line.strip('\n').split(',')))

    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(map(str, unique))
    return ngram_index_list



class bo_GPT(SingleObjectiveProblem):
    r"""The Prompt Gpt problem.
    Query the GPT model with a prompt and return the accuracy of the model on a given dataset.
    """
    n_constr: int = 0
    _optimal_value: float = 0.0

    def __init__(
            self,
            backend: str,
            dim: int,
            bounds: Optional[List] = None,
            maximize: Optional[bool] = False,
            args_selected: Optional[dict] = None,
            # a: Optional[float] = 20,
            # b: Optional[float] = 0.2,
            # c: Optional[float] = 2 * math.pi,
    ) -> None:
        super().__init__(backend=backend, dim=dim, bounds=bounds, maximize=maximize)
        
        self.args = parse_args(args_selected)
        args = self.args 
        ngram_list = pmi(args_selected)
        max_idx = len(ngram_list)-1
        if bounds is None:
            bounds = [(0, max_idx)] * args.prompt_length

        # specify a unique experiment_id for load_metric() otherwise will cause ERROR when having multiple run on a same server!
        task_name = args.task_name if args.task_name else args.train_file
        args.unique_task_name = task_name.replace("/", ".")
        args.experiment_id = random.randint(0, 1000000)

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
                if args.task_name in EXTERNAL_DATASET: # only sentiment
                    path_file_row = os.path.join(os.path.dirname(__file__), "dataset", args.task_name.lower() + ".json")
                    raw_datasets = create_dataset(path_file_row)
                else:
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

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

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
            template_base = template_cfg.replace('[MASK]', "<mask>") # Not use the tokenizer for gpt

            if sentence2_key:
                sent1_list = []
                for sent1 in examples[sentence1_key]:
                    sent1_list.append(sent1 + template_base)
                texts = (sent1_list, examples[sentence2_key])
            else:
                template = [template_base] * len(examples[sentence1_key])
                texts = (examples[sentence1_key], template)
            # result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True, add_special_tokens=False)
            result = {"input_ids": []}
            for text in texts:
                result["input_ids"].append(" ".join(text))

            texts = []
            template = [template_base] * len(examples[sentence1_key])
            if sentence2_key:
                for tuple_ in list(zip(examples[sentence1_key], template, examples[sentence2_key])):
                    sent_1 = tokenizer.tokenize(tuple_[0])[:200]
                    new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                    sent_2 = tokenizer.tokenize(tuple_[2])[:200]
                    new_sent_2 = tokenizer.convert_tokens_to_string(sent_2)
                    # texts.append(new_sent_1 + tokenizer.sep_token + new_sent_2 + tuple_[1])
                    texts.append(new_sent_1 + '<mask>' + new_sent_2 + tuple_[1])
                texts = tokenizer.convert_tokens_to_string(texts)
            else:
                for tuple_ in list(zip(examples[sentence1_key], template)):
                    sent_1 = tokenizer.tokenize(tuple_[0])[:400]
                    new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                    texts.append(new_sent_1 +" "+ tuple_[1])
            # result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result = {"input_ids": []}
            for text in texts:
                result["input_ids"].append("".join(text))

            if args.task_name:
                label_list = []
                for raw_label in examples["label"]:
                    label = LABEL_CONVERT[args.task_name][raw_label]
                    # target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                    # label_list.append(target_encodings[0])
                    label_list.append(str(label).lower())
                # result["labels"] = torch.tensor(label_list)
                result["labels"] = label_list
                    

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

        def process_dataset(dataset, preprocess_function, batch_size=100000, remove_columns=None, load_from_cache_file=False, desc="Running tokenizer on dataset"):
            return dataset.map(
                preprocess_function,
                batched=True,
                batch_size=batch_size,
                remove_columns=remove_columns,
                load_from_cache_file=load_from_cache_file,
                desc=desc,
            )

        with accelerator.main_process_first():
            remove_columns = raw_datasets["train"].column_names
            if args.k_shot >= 0: # k-shot learning value : 16
                # k-shot learning
                raw_train_dataset_split = raw_datasets["train"].train_test_split(test_size=0.5)
                train_dataset = process_dataset(raw_train_dataset_split['train'], preprocess_function_k_shot, remove_columns=remove_columns)
                eval_dataset = process_dataset(raw_train_dataset_split['test'], preprocess_function_k_shot, remove_columns=remove_columns)
                if args.task_name == 'mnli':
                    test_dataset = process_dataset(raw_datasets["validation_matched"], preprocess_function, remove_columns=remove_columns)
                    test_dataset_mm = process_dataset(raw_datasets["validation_mismatched"], preprocess_function, remove_columns=remove_columns)
                else:
                    test_dataset = process_dataset(raw_datasets["validation"], preprocess_function, remove_columns=remove_columns)
            else:
                train_dataset = process_dataset(raw_datasets["train"], preprocess_function, remove_columns=remove_columns)
                eval_dataset = process_dataset(raw_datasets["validation"], preprocess_function, remove_columns=remove_columns)
                test_dataset = process_dataset(raw_datasets["test" if args.file_name != None else "validation"], preprocess_function, remove_columns=remove_columns)

            #print("length of train data",len(train_dataset))
            #print("length of eval data",len(eval_dataset))
            #print("length of test data",len(test_dataset))  
            global train_dataset_example 
            train_dataset_example = train_dataset

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
        train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare( train_dataloader, eval_dataloader, test_dataloader)

        if args.task_name is not None and args.task_name not in EXTERNAL_DATASET:
            metric = load_metric("glue", str(args.experiment_id)) #, experiment_id=args.experiment_id)
        elif args.file_name in DOMAIN_DATASET:
            metric = load_metric('f1', str(args.experiment_id))
        else:
            metric = load_metric('accuracy', str(args.experiment_id), trust_remote_code=True)

        self.metric = metric
        self.logger = logging.getLogger(str(__name__))

        self.epoch = 0
        self.ngram_list = ngram_list
        max_idx = len(ngram_list)-1
        accelerator = Accelerator()

        
        # self._bounds = [[0, max_idx]] * args.prompt_length
        setattr(self, "_bounds", bounds)
        setattr(self, "bounds", bounds)
        setattr(self, "dim", args.prompt_length)
        setattr(self, "accelerator", accelerator)
        setattr(self, "ce_loss", CrossEntropyLoss())
        setattr(self, "train_dataloader", train_dataloader)
        setattr(self, "eval_dataloader", eval_dataloader)
        setattr(self, "test_dataloader", test_dataloader)
        setattr(self, "tokenizer", tokenizer)
        setattr(self, "config", config)
        setattr(self, "max_idx", max_idx)

    def _evaluate(self, x: List[List[float]]) -> np.ndarray:
        """
        Evaluate the model on the given input.

        Args:
            x (List[List[float]]): The input data.

        Returns:
            np.ndarray: The evaluated output.

        Raises:
            ValueError: If the input data is invalid.

        """
        back, kwargs = self._get_backend()
        # a, b, c = self.a, self.b, self.c
        # part1 = -a * back.exp(-b * back.sqrt((1 / self.dim) * back.sum(x * x, **kwargs)))
        # part2 = -back.exp((1 / self.dim) * back.sum(back.cos(c * x), **kwargs))
        # return (part1 + part2 + a + math.e).reshape(-1, 1)
        
        def evaluate_true(self, X, dataloader_type="train"): # funzione cge valuta | X tensor "list of list" token di input  | a tensor list of scores for each element in X

            # Get predicted prompt indices by taking argmax of probabilities
            # prompts_discrete_indices = X.argmax(1) 
            prompts_discrete_indices = X 
            args = self.args
            # model = self.model
            # tokenizer = self.tokenizer
            metric = self.metric
            max_length = {'train': 5, 'test': 300, 'eval': 5}
            epoch = self.epoch # NOT SURE IF THIS IS CORRECT WAY TO GET EPOCH, CHECK

            ngram_list = self.ngram_list
            accelerator = self.accelerator

            # If using n-gram prompts, convert indices to n-gram sequences
            if args.use_ngram:
                prompts_discrete_indices_ngram_list = []
                indices_list = prompts_discrete_indices.int().tolist()
                prompts_discrete_indices_ngram_list = [' ' + ngram_list[idx] if not ngram_list[idx].startswith(' ') else ngram_list[idx] for idx in indices_list]
                # prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list) fail to convert list of string in a tensor
                
                prompts_string = ''.join(prompts_discrete_indices_ngram_list)
            
            # Iterate through batches
            count_batch = 0
            if dataloader_type == "train":
                data_loader = self.train_dataloader

            elif dataloader_type == "test":
                print("dataloader: test")
                data_loader = self.test_dataloader
            else:
                print("dataloader: eval")
                data_loader = self.eval_dataloader
            
            test_batches = data_loader.dataset
            if len(test_batches) > max_length[dataloader_type]:
                test_batches = test_batches.select(range(0, max_length[dataloader_type]))
            row_response = []
            for step in range(len(test_batches)):
                count_batch += 1
                # Stop after 100 batches if trial run
                if args.trial and step >= 100:
                    break   

                input_x = test_batches['input_ids'][step]
                label_y = test_batches['labels'][step]

                # query_input = 'Definition: ' + prompts_string   + '\t' + input_x
                query_input = prompts_string   + '\t' + input_x
                response = chat_LLM( query_input, LABEL2ID_CONFIG[args.task_name])
                # return {"label": LABEL2ID_CONFIG[selected_label], "response":chat_completion}
                chat_completion = response['response']
                predictions = response['label']
                row_response.append(chat_completion)

                converted_target = LABEL2ID_CONFIG[args.task_name][label_y]

                # Update metrics
                metric.add_batch(
                predictions=accelerator.gather([predictions]),
                references=accelerator.gather([converted_target]),
                )
                
            # Compute overall metrics
            if args.file_name in DOMAIN_DATASET:
                eval_metric = metric.compute(average='macro')
            else:
                eval_metric = metric.compute()

            # print("*****\nprompt:", prompts_string, "\neval_metric:", eval_metric, "\nepoch:", epoch)

            # Extract key metric  
            if args.task_name == 'cola':
                key = 'matthews_correlation'
            elif (args.task_name in ['mnli', 'sst2', 'wnli', 'rte', 'qnli'] 
                            or args.file_name in ['MR', 'CR'] 
                            or args.task_name in EXTERNAL_DATASET):
                key = 'accuracy'
            else:
                key = 'f1'
            eval_result = eval_metric[key]

            # Append to results
            results.append(eval_result)

            # Log results
            global llm_logger
            if args.task_name not in llm_logger:
                llm_logger[args.task_name] = []
            current_eval = len(llm_logger[args.task_name]) + 1
            
            llm_logger[args.task_name].extend([{
                "current_eval" : current_eval,
                "chat_completion_responses" : row_response,
                "score" : eval_result,
                "prompt" : prompts_string,
                "total_tokens" : sum([c.usage.total_tokens for c in row_response])


            }])
            # Get the current date and time
            date_string = datetime.datetime.now().strftime("%m%d")
            # Define the file name
            file_name = f"{date_string}_llm_log_{args.task_name}.pickle"

            # Save llm_logger in a pickle file
            with open(file_name, "wb") as file:
                pickle.dump(llm_logger, file)
            return eval_result
        
        train_y = [ [evaluate_true(self, X=x_single.clone().detach())] for x_single in x ]
        
        back, _ = self._get_backend()
        
        try:
            train_y = back.array(train_y, dtype=np.float64)
        except:
            train_y = torch.tensor(train_y, dtype=torch.float64)
        
        return train_y.reshape(-1, 1)