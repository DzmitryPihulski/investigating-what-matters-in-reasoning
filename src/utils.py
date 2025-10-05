from datasets import Dataset
import pandas as pd
import os
import shutil
import yaml

def prepare_data(df: pd.DataFrame) -> Dataset:
    """
    Convert a pandas DataFrame with a 'messages' column (list of dicts)
    into a Hugging Face Dataset with conversational format.
    """
    conversations = []

    for _, row in df.iterrows():
        messages = row["messages"]
        for message in messages:
            if message['role'] == 'user':
                prompt = '[INST]' + message['content'] + '[/INST]'
            elif message['role'] == 'assistant':
                completion = message['content']
            else:
                raise ValueError("Invalid message format")

        conversations.append({"prompt": prompt, "completion": completion})

    return Dataset.from_list(conversations)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def copy_files_to_folder(folder_path: str, *file_paths: str):
    for file_path in file_paths:
        if os.path.isfile(file_path):
            shutil.copy(file_path, folder_path)
        else:
            print(f"Warning: File '{file_path}' does not exist and will not be copied.")

