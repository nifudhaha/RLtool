import os
import json
import time
import base64
import argparse
from tqdm import tqdm
import threading
import concurrent.futures
import signal
import sys
import re
from PIL import Image
from omegaconf import DictConfig
from datasets import load_dataset, Dataset
from openai import OpenAI
from typing import List, Tuple, Dict, Any, Optional, Callable
import io
import pandas as pd

sys.path.append('..')

def verify(response, answer):
    """验证响应是否正确"""
    if not response:
        return False
    if answer.startswith('('):
        answer = answer[1]
    match = re.search(r'\\boxed\{(.*)\}', response, re.DOTALL)
    if match:
        return match.group(1).strip().upper() == answer.strip().upper()
    return False

class DatasetEvaluator:
    def __init__(self, model_name: str, port_pool: List[int], workers: int, isremote: bool, instruction_following: str, evaluate: bool = False):
        self.model_name = model_name
        self.port_pool = port_pool
        self.workers = workers
        self.isremote = isremote
        self.instruction_following = instruction_following
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.evaluate = evaluate
        
        # 端口轮询相关
        self.port_lock = threading.Lock()
        self.port_index = 0
        
        # 文件写入锁
        self.file_lock = threading.Lock()
        
        # 线程池执行器
        self.executor = None
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n正在优雅地关闭程序，请稍候...")
        if self.executor is not None:
            self.executor.shutdown(wait=False)
        sys.exit(0)

    def get_next_port(self) -> int:
        """从端口池中轮询获取下一个可用端口"""
        with self.port_lock:
            current_port = self.port_pool[self.port_index]
            self.port_index = (self.port_index + 1) % len(self.port_pool)
            return current_port

    def save_base64_image(self, base64_str: str, image_dir: str, sample_id: int, image_index: int) -> str:
        """Process base64 image and save to file"""
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        
        image_data = base64.b64decode(base64_str)
        image_format = "png"
        filename = f"{sample_id}_{image_index}.{image_format}"
        filepath = os.path.join(image_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        return filename

    def process_conversation_images(self, conversation: List[Dict], image_dir: str, sample_id: int) -> List[Dict]:
        """Process images in conversation"""
        new_conversation = []
        image_index = 0
        
        for message in conversation:
            new_message = message.copy()
            
            if isinstance(message.get('content'), list):
                new_content = []
                for item in message['content']:
                    if item.get('type') == 'image_url' and 'image_url' in item:
                        url = item['image_url'].get('url', '')
                        if url.startswith('data:image'):
                            filename = self.save_base64_image(url, image_dir, sample_id, image_index)
                            image_index += 1
                            new_content.append({
                                'type': 'image_url',
                                'image_url': {'url': f"{image_dir}/{filename}"}
                            })
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                new_message['content'] = new_content
            
            new_conversation.append(new_message)
        
        return new_conversation

    def create_agent(self, port: int):
        """创建一个Agent实例"""
        if self.isremote:
            agent_config = DictConfig({
                "max_turns": 5,
                "max_tokens_per_turn": 4096,
                "save_intermediate_responses": True,
                "openai_api_key": "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",
                "openai_model": self.model_name,
                "openai_temperature": 0.5,
            })
        else:
            agent_config = DictConfig({
                "max_turns": 5,
                "max_tokens_per_turn": 2048,
                "save_intermediate_responses": True,
                "openai_api_base": f"http://localhost:{port}/v1/chat/completions",
                "openai_api_key": "fake-api-key",
                "openai_model": self.model_name,
                "openai_temperature": 0.2,
            })

        from verl.api_agent_new import Agent
        return Agent(agent_config)

    def _base_process_sample(self, idx: int, prompt: str, images: List[Image.Image], 
                           image_dir: str, output_path: str, result_data: Dict[str, Any]) -> Tuple[int, bool]:
        """基础样本处理函数，包含公共逻辑"""
        try:
            port = self.get_next_port()
            
            agent = self.create_agent(port)
            response, conversation = agent.chat_with_tools(
                system_prompt=self.instruction_following,
                prompt=prompt,
                images=images
            )
            
            conversation = self.process_conversation_images(conversation, image_dir, idx)
            
            # 构建基础结果
            result = {
                'sample_id': idx,
                'response': response,
                'conversation': conversation,
                **result_data  # 合并传入的结果数据
            }
            
            # 如果需要评估，添加验证结果
            if self.evaluate and 'answer' in result_data:
                result['is_correct'] = verify(response, result_data['answer'])
            
            # 写入文件
            with self.file_lock:
                with open(output_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            
            return idx, True
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return idx, False

    def process_cvbench_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理CV-Bench样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = '<image>' + sample['prompt']
        images = [sample['image']]
        
        result_data = {
            'type': sample['type'],
            'task': sample['task'],
            'question': sample['question'],
            'choices': sample['choices'],
            'answer': sample['answer'],
            'metadata': {
                'base_image_dir': image_dir,
                'filename': sample.get('filename', ''),
                'source': sample.get('source', ''),
                'source_dataset': sample.get('source_dataset', '')
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_mmstar_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理MMStar样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = sample['question']
        images = [sample['image']]
        
        result_data = {
            'question': sample['question'],
            'answer': sample['answer'],
            'category': sample.get('category', ''),
            'l2_category': sample.get('l2_category', ''),
            'metadata': {
                'base_image_dir': image_dir,
                'index': sample.get('index', idx),
                'meta_info': sample.get('meta_info', {})
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_blink_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理BLINK样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = '<image>' + sample['prompt']
        images = [sample['image_1']]
        
        result_data = {
            'question': sample['question'],
            'sub_task': sample['sub_task'],
            'choices': sample['choices'],
            'answer': sample['answer'],
            'metadata': {
                'base_image_dir': image_dir,
                'idx': sample.get('idx', '')
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_sat_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理SAT样本"""
        idx, sample, image_dir, output_path = args
        
        sample_question = sample['processed_mcqa']['question']
        sample_options = sample['processed_mcqa']['options']
        sample_answer_idx = sample['processed_mcqa']['answer']
        image_path = sample['images'][0]
        
        sample_image = Image.open(image_path)
        
        options_text = "Answer from the following choices:"
        for i, option in enumerate(sample_options):
            options_text += f"\n({chr(65+i)}) {option}"
        
        prompt = f"{sample_question}{options_text}\nThe answer is {chr(65 + sample_answer_idx)}"
        images = [sample_image]
        
        result_data = {
            'question': sample_question,
            'options': sample_options,
            'answer': chr(65 + sample_answer_idx),
            'metadata': {
                'base_image_dir': image_dir,
                'id': sample.get('id', '')
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_mmvp_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理MMVP样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = sample['Question'] + f"\nAnswer from the following options:\n{sample['Options']}"
        images = sample['image']
        
        result_data = {
            'question': sample['Question'],
            'choices': sample['Options'],
            'answer': sample['Correct Answer'],
            'metadata': {
                'base_image_dir': image_dir,
                'filename': sample.get('filename', ''),
                'source': sample.get('source', ''),
                'source_dataset': sample.get('source_dataset', '')
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_natural_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理NaturalBench样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = sample['question']
        if sample['question_type'] == 'yes_no':
            prompt += '\nSelect from the following options:\n(A) Yes\n(B) No'
            answer = 'A' if sample['answer'] == 'Yes' else 'B'
        elif sample['question_type'] == 'multiple_choice':
            prompt += 'Please output the letter corresponding to the correct option.'
            answer = sample['answer']
        else:
            answer = sample['answer']
        
        prompt = '<image>' + prompt
        images = [sample['image']]
        
        result_data = {
            'question': sample['question'],
            'answer': answer,
            'metadata': {
                'base_image_dir': image_dir,
                'idx': sample.get('index', '')
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_blink_hard_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理BLINK Hard样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = '<image>' + sample['prompt']
        images = [Image.open(io.BytesIO(sample['image_bytes']))]
        
        result_data = {
            'question': sample['prompt'],
            'category': sample['category'],
            'answer': sample['answer'],
            'metadata': {
                'base_image_dir': image_dir
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_tallyqa_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理TallyQA样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = '<image>' + sample['question'] + '\nPut the number of the answer in \\boxed{}.'
        images = [Image.open(sample['image'])]
        
        result_data = {
            'question': sample['question'],
            'answer': sample['answer']
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_aokvqa_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理AOKVQA样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = sample['question'] + '\nAnswer from the following choices:\n'
        for i, choice in enumerate(sample['choices']):
            prompt += f"({chr(65+i)}) {choice}\n"
        prompt = '<image>' + prompt + 'Your answer should include \\boxed{answer letter}.'
        
        images = [sample['image']]
        
        result_data = {
            'question': sample['question'],
            'answer': chr(65+sample['correct_choice_idx'])
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_mmmu_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理MMMU样本"""
        idx, sample, image_dir, output_path = args
        
        # 获取样本数据
        sample_question = sample['question']
        from ast import literal_eval
        sample_options = literal_eval(sample['options'])
        sample_answer = sample['answer']
        sample_images = []
        
        # 处理图像 - MMMU可能有多个图像
        for i in range(1, 8):  # MMMU最多可以有7个图像
            img_key = f'image_{i}'
            if img_key in sample and sample[img_key] is not None:
                sample_images.append(sample[img_key])
        
        # 如果没有编号图像，检查'image'键
        if not sample_images and 'image' in sample and sample['image'] is not None:
            sample_images.append(sample['image'])
        
        # 构建带选项的提示
        prompt = sample_question
        if sample_options:
            prompt += "\nPlease select from the following options:\n"
            for i, option in enumerate(sample_options):
                prompt += f"({chr(65+i)}) {option}\n"
        
        # 如果存在图像，添加图像占位符
        if sample_images:
            image_placeholder = '<image>' * len(sample_images)
            prompt = image_placeholder + prompt
        
        result_data = {
            'question': sample_question,
            'options': sample_options,
            'answer': sample_answer
        }
        
        return self._base_process_sample(idx, prompt, sample_images if sample_images else None, 
                                       image_dir, output_path, result_data)

    def process_mmbench_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理MMBench样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = '<image>' + sample['question']
        images = [Image.open(io.BytesIO(base64.b64decode(sample['image'])))]
        
        # 构建选项字典
        options = ['A', 'B', 'C', 'D']
        options_dict = {opt: sample.get(opt, '') for opt in options if sample.get(opt)}
        prompt += "\nAnswer from the following options:"
        for k,v in options_dict.items():
            prompt += f"\n({k}) {v}"
        
        result_data = {
            'question': sample['question'],
            'hint': sample.get('hint', ''),
            'options': options_dict,
            'answer': sample['answer'],
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_mmvet_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理MM-Vet样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = '<image>' + sample['question']
        images = [sample['image']]
        
        result_data = {
            'question': sample['question'],
            'answer': sample['answer'],
            'capability': sample.get('capability', []),
            'metadata': {
                'base_image_dir': image_dir,
                'imagename': sample.get('imagename', ''),
                'category': sample.get('category', '')
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)
    
    def process_mme_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理MME样本"""
        idx, sample, image_dir, output_path = args
        
        prompt = '<image>' + sample['question']
        images = [sample['image']]
        
        result_data = {
            'question': sample['question'],
            'answer': sample['answer'],
            'category': sample.get('category', ''),
            'metadata': {
                'base_image_dir': image_dir,
                'image_path': sample.get('image_path', ''),
                'question_id': sample.get('question_id', idx)
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_realworldqa_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理RealworldQA样本"""
        idx, sample, image_dir, output_path = args

        prompt = '<image>' + sample['question']
        # RealworldQA图片字段假设为'image'，如有不同请调整
        images = [sample['image']]

        result_data = {
            'question': sample['question'],
            'answer': sample['answer'],
            'metadata': {
                'base_image_dir': image_dir,
                'id': sample.get('id', idx)
            }
        }

        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def process_mathvista_sample(self, args: Tuple) -> Tuple[int, bool]:
        """处理MathVista样本"""
        idx, sample, image_dir, output_path = args
        
        # 构建提示，MathVista通常包含问题和可能的选项
        prompt = '<image>' + sample['question']
        
        # 添加选项
        if 'choices' in sample and sample['choices']:
            prompt += '\nAnswer from the following options:\n'
            for i, choice in enumerate(sample['choices']):
                prompt += f'{chr(65+i)}. {choice}\n'

        images = [sample['decoded_image']]
        
        result_data = {
            'question': sample['question'],
            'answer': chr(65 + sample['choices'].index(sample['answer'])) if sample['choices'] else sample['answer'],
            'metadata': {
                'base_image_dir': image_dir,
            }
        }
        
        return self._base_process_sample(idx, prompt, images, image_dir, output_path, result_data)

    def run_parallel_evaluation(self, args_list: List[Tuple], process_func: Callable) -> Tuple[int, int]:
        """运行并行评估"""
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        try:
            results = list(tqdm(self.executor.map(process_func, args_list), total=len(args_list)))
        finally:
            self.executor.shutdown()
            self.executor = None
        
        success_count = sum(1 for _, success in results if success)
        fail_count = len(results) - success_count
        return success_count, fail_count

    def _base_evaluate(self, dataset_name: str, dataset_loader: Callable, process_func: Callable, 
                      dataset_args: Optional[Dict] = None):
        """基础评估函数，包含公共的评估流程"""
        # 设置分层输出路径：evaluation/model/benchmark
        base_dir = f"{self.model_name}/{dataset_name}"
        output_path = f"{base_dir}/results_{self.timestamp}.jsonl"
        image_dir = f"{base_dir}/images_{self.timestamp}"
        
        # 创建分层目录结构
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # 创建空文件
        if not os.path.exists(output_path):
            with open(output_path, 'w') as f:
                pass
        
        # 加载数据集
        print(f"Loading {dataset_name} dataset...")
        if dataset_args:
            dataset = dataset_loader(**dataset_args)
        else:
            dataset = dataset_loader()
        
        print(f"Starting evaluation of {len(dataset)} {dataset_name} samples using {self.workers} parallel worker threads")
        
        # 准备参数列表
        args_list = [(idx, sample, image_dir, output_path) 
                    for idx, sample in enumerate(dataset)]
        
        # 运行评估
        success_count, fail_count = self.run_parallel_evaluation(args_list, process_func)
        
        print(f"{dataset_name} evaluation complete, results saved to {output_path}")
        print(f"Successfully processed: {success_count} samples, Failed: {fail_count} samples")

    # 各数据集的评估方法
    def evaluate_cvbench(self):
        def load_cvbench():
            ds = load_dataset("nyu-visionx/CV-Bench", "default")
            return ds["test"]
        
        self._base_evaluate("cvbench", load_cvbench, self.process_cvbench_sample)

    def evaluate_mmstar(self):
        def load_mmstar():
            mmstar_dataset = load_dataset("Lin-Chen/MMStar")
            return mmstar_dataset["val"]
        
        self._base_evaluate("mmstar", load_mmstar, self.process_mmstar_sample)

    def evaluate_blink(self):
        def load_blink():
            selected_subtasks = ['Counting', 'Relative_Depth', 'Spatial_Relation']
            all_test_samples = []
            for subtask in selected_subtasks:
                print(f"Loading BLINK subtask: {subtask}")
                subtask_dataset = load_dataset("BLINK-Benchmark/BLINK", subtask)
                subtask_test_samples = subtask_dataset["val"]
                all_test_samples.extend(subtask_test_samples)
            return all_test_samples
        
        self._base_evaluate("blink", load_blink, self.process_blink_sample)

    def evaluate_sat(self):
        def load_sat():
            with open('/mnt/gold/cdp/zzt/datasets/sat/sft_remain.jsonl', 'r') as f:
                sat_dataset = [json.loads(line) for line in f]
            return sat_dataset[:1000]
        
        self._base_evaluate("sat", load_sat, self.process_sat_sample)

    def evaluate_mmvp(self):
        def load_mmvp():
            mmvp_dataset = load_dataset("parquet", data_files="../data/mmvp_val.parquet")
            return mmvp_dataset["train"]
        
        self._base_evaluate("mmvp", load_mmvp, self.process_mmvp_sample)

    def evaluate_naturalbench(self):
        def load_naturalbench():
            natural_dataset = load_dataset("BaiqiL/NaturalBench")
            natural_train_dataset = []
            
            for item in natural_dataset["train"]:
                for i in range(0, 2):
                    for j in range(0, 2):
                        natural_train_dataset.append({
                            'index': item['Index']*4+i*2+j,
                            'image': item[f"Image_{i}"],
                            'question': item[f"Question_{j}"],
                            'question_type': item['Question_Type'],
                            'answer': item[f"Image_{i}_Question_{j}"]
                        })
            return natural_train_dataset
        
        self._base_evaluate("naturalbench", load_naturalbench, self.process_natural_sample)

    def evaluate_blink_hard(self):
        def load_blink_hard():
            blink_hard_dataset = load_dataset("parquet", data_files="../data/blink_hard_dataset.parquet")
            return blink_hard_dataset["train"]
        
        self._base_evaluate("blink-hard", load_blink_hard, self.process_blink_hard_sample)

    def evaluate_tallyqa(self):
        def load_tallyqa():
            with open('/mnt/gold/cdp/zzt/datasets/tallyqa/tallyqa_vg.json') as f:
                return json.load(f)
        
        self._base_evaluate("tallyqa", load_tallyqa, self.process_tallyqa_sample)

    def evaluate_aokvqa(self):
        def load_aokvqa():
            return load_dataset('parquet', data_files=[
                '/mnt/gold/cdp/zzt/datasets/aokvqa/data/train0.parquet', 
                '/mnt/gold/cdp/zzt/datasets/aokvqa/data/train1.parquet'
            ], split='train')
        
        self._base_evaluate("aokvqa", load_aokvqa, self.process_aokvqa_sample)

    def evaluate_mmmu(self):
        def load_mmmu():
            mmmu_val_dataset = []
            configs = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 
                      'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 
                      'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 
                      'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 
                      'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 
                      'Physics', 'Psychology', 'Public_Health', 'Sociology']
            
            for config in configs:
                config_dataset = load_dataset("MMMU/MMMU", config)
                mmmu_val_dataset.extend(config_dataset["validation"])
                print(f"Loaded {len(config_dataset['validation'])} samples from {config} configuration")
            return mmmu_val_dataset
        
        self._base_evaluate("mmmu", load_mmmu, self.process_mmmu_sample)

    def evaluate_mmbench(self):
        def load_mmbench():
            return Dataset.from_pandas(pd.read_csv('MMBench_DEV_EN_legacy.tsv', sep='\t'))
        
        self._base_evaluate("mmbench", load_mmbench, self.process_mmbench_sample)

    def evaluate_mmvet(self):
        def load_mmvet():
            mmvet_dataset = load_dataset("whyu/MM-Vet")
            return mmvet_dataset["test"]
        
        self._base_evaluate("mmvet", load_mmvet, self.process_mmvet_sample)
        
    def evaluate_mme(self):
        def load_mme():
            mme_dataset = load_dataset("darkyarding/MME")
            return mme_dataset["test"]
        
        self._base_evaluate("mme", load_mme, self.process_mme_sample)

    def evaluate_realworldqa(self):
        def load_realworldqa():
            # 假设数据集已上传到HF datasets，或本地parquet/jsonl文件
            # 下面以parquet为例，如有不同请调整
            realworldqa_dataset = load_dataset("lmms-lab/RealWorldQA")
            return realworldqa_dataset["test"]
        self._base_evaluate("realworldqa", load_realworldqa, self.process_realworldqa_sample)

    def evaluate_mathvista(self):
        def load_mathvista():
            mathvista_dataset = load_dataset("AI4Math/MathVista")
            return mathvista_dataset["testmini"]
        
        self._base_evaluate("mathvista", load_mathvista, self.process_mathvista_sample)


def get_system_prompt(prompt_type: str) -> str:
    """获取系统提示"""
    if prompt_type == 'agent':
        from data.prompt_g_d import get_system_prompt
        return get_system_prompt()
    elif prompt_type == 'text':
        from data.prompt_text import get_system_prompt
        return get_system_prompt()
    elif prompt_type == 'none':
        return None
    elif prompt_type == 'agent_api':
        from data.prompt_g import get_system_prompt
        return get_system_prompt()
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def main():
    parser = argparse.ArgumentParser(description='Run dataset evaluation')
    parser.add_argument('--model-name', type=str, default='Qwen2.5-VL', help='Model name')
    parser.add_argument('--port-pool', type=str, default='8015,8016,8017,8018', help='API server port pool, comma separated')
    parser.add_argument('--workers', type=int, default=32, help='Number of parallel worker threads')
    parser.add_argument('--dataset', type=str, default='cvbench', help='Dataset to evaluate')
    parser.add_argument('--remote', action='store_true', help='Use remote API')
    parser.add_argument('--prompt', type=str, default='agent', help='Prompt type')
    parser.add_argument('--evaluate', action='store_true', help='Enable answer verification')
    args = parser.parse_args()

    # 解析端口池
    port_pool = [int(port.strip()) for port in args.port_pool.split(',')]
    print(f"使用端口池: {port_pool}")
    
    # 获取系统提示
    try:
        instruction_following = get_system_prompt(args.prompt)
        print(f"System prompt: {instruction_following}")
    except ValueError as e:
        print(str(e))
        return
    
    # 创建评估器
    evaluator = DatasetEvaluator(
        model_name=args.model_name,
        port_pool=port_pool,
        workers=args.workers,
        isremote=args.remote,
        instruction_following=instruction_following,
        evaluate=args.evaluate
    )
    
    # 数据集评估方法映射
    dataset_methods = {
        'cvbench': evaluator.evaluate_cvbench,
        'mmstar': evaluator.evaluate_mmstar,
        'blink': evaluator.evaluate_blink,
        'sat': evaluator.evaluate_sat,
        'naturalbench': evaluator.evaluate_naturalbench,
        'blink-hard': evaluator.evaluate_blink_hard,
        'mmvp': evaluator.evaluate_mmvp,
        'tallyqa': evaluator.evaluate_tallyqa,
        'aokvqa': evaluator.evaluate_aokvqa,
        'mmmu': evaluator.evaluate_mmmu,
        'mmbench': evaluator.evaluate_mmbench,
        'mmvet': evaluator.evaluate_mmvet,
        'mme': evaluator.evaluate_mme,
        'realworldqa': evaluator.evaluate_realworldqa,
        'mathvista': evaluator.evaluate_mathvista,  
    }
    
    # 运行对应的评估方法
    if args.dataset in dataset_methods:
        dataset_methods[args.dataset]()
    else:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available datasets: {', '.join(dataset_methods.keys())}")
        return


if __name__ == "__main__":
    main()