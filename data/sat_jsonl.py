import argparse
import os
import random

import json
import polars as pl
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
from PIL import Image
from prompt import *
import sys


def process_dataset(dataset, split_name, ability, instruction_following, data_source):
    """Common function to process dataset"""
    processed_data = []
    for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {split_name} set"):
        item = row["processed_mcqa"]
        prompt = item["question"]
        options = item["options"]
        
        if options:
            answer = chr(65 + item["answer"])
            prompt += f"\nAnswer from the following choices:"
            for i, option in enumerate(options):
                prompt += f"\n({chr(65 + i)}) {option}"
            prompt += '\nPut answer letter in the \\boxed{}'
        else:
            answer = item["answer"]
            
        images = row["images"]
        images = [f"file://{img}" for img in images]

        processed_data.append({
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": instruction_following,
                },
                {
                    "role": "user",
                    "content": '<image>'*len(images) + prompt,
                }
            ],
            "images": [{'image': img} for img in images],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split_name,
                "index": idx, 
                "question": prompt,
            },
        })
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/geo3k")
    parser.add_argument("--prompt", default="agent")

    args = parser.parse_args()
    
    # Get system prompt
    if args.prompt == 'agent':
        system_prompt = RL_PROMPT
    elif args.prompt == 'text':
        system_prompt = TEXT_RL_PROMPT
    elif args.prompt == 'none':
        system_prompt = None
    elif args.prompt == 'agent_api':
        system_prompt = SYN_PROMPT
    else:
        print(f"Unknown prompt type: {args.prompt}")
        exit(1)


    data_source = "sat"

    jsons = [
        'rl/rl_data.json'
    ]
    dataset = []
    
    for json_file in jsons:
        with open(json_file, 'r') as f:
            if json_file.endswith('.jsonl'):
                for line in f:
                    data = json.loads(line)
                    dataset.append(data)
            elif json_file.endswith('.json'):
                data = json.load(f)
                dataset.extend(data)
        
    dataset_size = len(dataset)
    
    # Shuffle dataset indices
    random.seed(42)
    random.shuffle(dataset)

    # Split dataset using indices
    train_dataset = dataset[1000:]
    test_dataset = dataset[:1000]

    # Process training set
    train_processed = process_dataset(train_dataset, "train", "other", system_prompt, data_source)
    test_processed = process_dataset(test_dataset, "test", "other", system_prompt, data_source)
    

    # Convert processed data to polars DataFrame
    train_df = pl.from_dicts(train_processed)
    test_df = pl.from_dicts(test_processed)

    local_dir = os.path.expanduser(args.local_dir) # Expand ~ to home directory
    os.makedirs(local_dir, exist_ok=True) # Ensure local directory exists

    # Split train into three shards
    num_shards = 1
    shard_size = len(train_df) // num_shards
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_shards - 1 else len(train_df)
        shard_df = train_df[start:end]
        shard_path = os.path.join(local_dir, f"train{i}.parquet")
        print(f"Writing train shard {i} to {shard_path}")
        shard_df.write_parquet(shard_path)

    test_path = os.path.join(local_dir, "test.parquet")
    print(f"Writing test set to {test_path}")
    test_df.write_parquet(test_path)

    print("Processing finished.")