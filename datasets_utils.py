import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

TASKS_IMPLEMENTED = ['sentiment']

def create_dataset(path_file_row: str)-> DatasetDict:
    task_name = path_file_row.replace("/", ".").split('.')[-2]
    if task_name not in TASKS_IMPLEMENTED:
        raise ValueError(f"Task {task_name} not implemented. Choose one of {TASKS_IMPLEMENTED}")
    
    # Load the dataset
    df = pd.read_json(path_file_row)
    df['input'] = df['examples'].apply(lambda x: x['input'] if isinstance(x, dict) else None)
    df['label'] = df['examples'].apply(lambda x: 0 if isinstance(x, dict) and x['output'] == 'negative' 
                                                    else 1 if isinstance(x, dict) and x['output'] == 'positive' else None)
    df = df.dropna(subset=['input']) # 1 row dropped in sentiment.json

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

# Usage
path_file_row = r'/home/vscode/Black-Box-Prompt-Learning/dataset/sentiment.json'
dataset = create_dataset(path_file_row)
print(dataset)