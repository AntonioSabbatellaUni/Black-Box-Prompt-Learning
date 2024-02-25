import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

TASKS_IMPLEMENTED = ['sentiment', 'sentence_similarity', 'word_in_context']


def process_dataframe(df: pd.DataFrame, task_name: str):
    df['input'] = df['examples'].apply(lambda x: x['input'] if isinstance(x, dict) else None)
    
    label_mapping = {
        'sentiment': {
            'negative': 0,
            'positive': 1
        },
        'sentence_similarity': {
            '0 - definitely not': 0,
            '1 - probably not': 1,
            '2 - possibly': 2,
            '3 - probably': 3,
            '4 - almost perfectly': 4,
            '5 - perfectly': 5
        },
        'word_in_context': {
            'not the same': 0,
            'no': 0,
            'false': 0,
            'same': 1,
            'yes': 1,
            'true': 1
        }
    }

    if task_name in label_mapping:
        df['label'] = df['examples'].apply(lambda x: label_mapping[task_name].get(x['output']) if isinstance(x, dict) else None)
        df = df.dropna(subset=['input'])
    else:
        raise ValueError(f"Task {task_name} not label-mapped. Choose one of {TASKS_IMPLEMENTED}")
    return df

def create_dataset(path_file_row: str)-> DatasetDict:
    task_name = path_file_row.replace("/", ".").split('.')[-2]
    if task_name not in TASKS_IMPLEMENTED:
        raise ValueError(f"Task {task_name} not implemented. Choose one of {TASKS_IMPLEMENTED}")
    
    # # Load the dataset
    df = pd.read_json(path_file_row)

    df = process_dataframe(df, task_name)

    # Dataframe split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_end = int(len(df) * 0.50)
    val_end = int(len(df) * 0.75)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Create DatasetDict objects for each set
    train_dataset = Dataset.from_dict({
        "sentence": train_df["input"],
        "label": train_df["label"]
    })
    val_dataset = Dataset.from_dict({
        "sentence": val_df["input"],
        "label": val_df["label"]
    })
    test_dataset = Dataset.from_dict({
        "sentence": test_df["input"],
        "label": test_df["label"]
    })

    # Create the final DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    return dataset

# # Usage
# path_file_row = r'/home/vscode/Black-Box-Prompt-Learning/dataset/sentence_similarity.json'
# path_file_row = r'/home/vscode/Black-Box-Prompt-Learning/dataset/sentiment.json'
# path_file_row = r'/home/vscode/Black-Box-Prompt-Learning/dataset/word_in_context.json'
# dataset = create_dataset(path_file_row)
# print(dataset)

