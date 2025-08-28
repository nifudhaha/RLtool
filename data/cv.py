import argparse
import os

import datasets
from datasets import Dataset

from qwen_vl_utils import smart_resize

import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/geo3k")
    parser.add_argument("--prompt", default="agent")

    args = parser.parse_args()
    
    # Get system prompt
    if args.prompt == 'agent':
        from prompt_g_d import get_system_prompt
        system_prompt = get_system_prompt()
    elif args.prompt == 'text':
        from prompt_text import get_system_prompt
        system_prompt = get_system_prompt()
    elif args.prompt == 'none':
        system_prompt = None
    elif args.prompt == 'agent_api':
        from prompt_g import get_system_prompt
        system_prompt = get_system_prompt()
    else:
        print(f"Unknown prompt type: {args.prompt}, using default agent prompt")
        exit(1)

    data_source = "nyu-visionx/CV-Bench"

    dataset = datasets.load_dataset(data_source)
    # dataset = dataset.filter(lambda x: x['task'] == 'Depth')

    # train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    # test_dataset = Dataset.from_dict(test_dataset[:100])
    
    # ratio = 0.8
    # dataset_size = len(test_dataset)
    # train_size = int(dataset_size * ratio)
    
    # # 随机分割数据集
    # # 先将数据集打乱
    shuffled_dataset = test_dataset.shuffle(seed=42)
    
    # # 分割成训练集和测试集
    # # train_dataset = Dataset.from_dict(shuffled_dataset[:train_size])
    test_dataset = Dataset.from_dict(shuffled_dataset[:])

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("question")
            prompt = example.pop("prompt")
            answer = example.pop("answer")
            images = example.pop("image")
            if not isinstance(images, list):
                images = [images]
            
            for i, img in enumerate(images):
                h,w = smart_resize(img.size[1], img.size[0], max_pixels=1500*28*28)
                images[i] = img.resize((w, h))

            data = {
                "data_source": data_source,
                "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": '<image>' + prompt + '\nPut answer letter in the \\boxed{}',
                }
                ],
                "images": images,
                "ability": "depth",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": prompt,
                },
            }
            return data

        return process_fn

    # train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir

    # train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))