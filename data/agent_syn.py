import os
import json
import time
import base64
import argparse
import random
import threading
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm
from PIL import Image

from verl.api_agent import Agent
from prompt_g import get_system_prompt


class AgentSynthesizer:
    """Agent synthesis processor for visual question answering tasks."""
    
    def __init__(self, config_file: str, max_workers: int = 16):
        self.config_file = config_file
        self.max_workers = max_workers
        self.file_lock = threading.Lock()
        
        # Initialize paths and directories
        self._setup_paths()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Initialize agent
        self.agent = self._create_agent()
    
    def _setup_paths(self) -> None:
        """Setup output paths and directories."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.output_path = f"syn/{timestamp}.jsonl"
        self.image_dir = f"syn/{timestamp}_images"

        # Create directories
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        
        # Ensure output file exists
        if not Path(self.output_path).exists():
            Path(self.output_path).touch()
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from config file."""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def _create_agent(self) -> Agent:
        """Create and configure the Agent."""
        agent_config = DictConfig({
            "max_turns": 5,
            "max_tokens_per_turn": 2048,
            "save_intermediate_responses": True,
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": "gpt-4.1",
            "openai_temperature": 1.0,
        })
        return Agent(agent_config)
    
    def _save_base64_image(self, base64_str: str, sample_id: int, image_index: int) -> str:
        """Save base64 image to file and return filename."""
        # Extract base64 string from data URL
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        
        # Decode and save image
        image_data = base64.b64decode(base64_str)
        filename = f"{sample_id}_{image_index}.png"
        filepath = Path(self.image_dir) / filename
        
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        return filename
    
    def _process_conversation_images(self, conversation: List[Dict], sample_id: int) -> List[Dict]:
        """Process and save images in conversation, replacing base64 with file paths."""
        new_conversation = []
        image_index = 0
        
        for message in conversation:
            new_message = message.copy()
            
            if isinstance(message.get('content'), list):
                new_content = []
                for item in message['content']:
                    if (item.get('type') == 'image_url' and 
                        'image_url' in item and 
                        item['image_url'].get('url', '').startswith('data:image')):
                        
                        # Save base64 image and replace with file path
                        filename = self._save_base64_image(
                            item['image_url']['url'], sample_id, image_index
                        )
                        image_index += 1
                        
                        new_content.append({
                            'type': 'image_url',
                            'image_url': {'url': f"{self.image_dir}/{filename}"}
                        })
                    else:
                        new_content.append(item)
                new_message['content'] = new_content
            
            new_conversation.append(new_message)
        
        return new_conversation
    
    def _prepare_prompt(self, qa: Dict[str, Any]) -> Tuple[str, str]:
        """Prepare prompt and expected answer from QA data."""
        prompt = qa['question']
        options = qa.get('options', [])
        
        if options:
            answer = qa['options'][qa['answer']]
            prompt += f"\nAnswer from the following choices:"
            for i, option in enumerate(options):
                prompt += f"\n{chr(65 + i)}. {option}"
            prompt += '\nPut answer letter in the \\boxed{}'
            
            # Convert answer to letter format
            expected_answer = chr(65 + options.index(answer))
        else:
            answer = qa['answer']
            expected_answer = answer
        
        return prompt, expected_answer
    
    def _process_single_sample(self, args: Tuple[int, Dict, Agent, str]) -> Tuple[int, bool]:
        """Process a single sample through the agent."""
        idx, sample, agent, image_dir = args
        
        try:
            metadata = sample['metadata']
            
            # Load image and prepare QA
            sample_image = Image.open(metadata['images'][0])
            qa = metadata['processed_mcqa']
            
            # Prepare prompt and answer
            sample_prompt, expected_answer = self._prepare_prompt(qa)
            
            # Get agent response
            response, conversation = agent.chat_with_tools(
                system_prompt=get_system_prompt(),
                prompt='<image>' + sample_prompt,
                images=[sample_image]
            )
            
            # Process conversation images
            processed_conversation = self._process_conversation_images(conversation, idx)
            
            # Prepare result
            result = {
                'sample_id': idx,
                'question': qa['question'],
                'answer': expected_answer,
                'response': response,
                'conversation': processed_conversation,
                'metadata': metadata
            }
            
            # Thread-safe file writing
            self._save_result(result)
            
            return idx, True
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return idx, False
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """Save result to output file in thread-safe manner."""
        with self.file_lock:
            with open(self.output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    def run_synthesis(self) -> None:
        """Run the complete synthesis process."""
        print(f"Starting synthesis of {len(self.dataset)} samples using {self.max_workers} workers")
        
        # Prepare arguments for parallel processing
        args_list = [
            (idx, sample, self.agent, self.image_dir)
            for idx, sample in enumerate(self.dataset)
        ]
        
        # Shuffle for better load distribution
        random.seed(42)
        random.shuffle(args_list)
        
        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self._process_single_sample, args_list),
                total=len(args_list)
            ))
        
        # Report results
        self._report_results(results)
    
    def _report_results(self, results: List[Tuple[int, bool]]) -> None:
        """Report processing results."""
        success_count = sum(1 for _, success in results if success)
        fail_count = len(results) - success_count
        
        print(f"Synthesis complete, results saved to {self.output_path}")
        print(f"Successfully processed: {success_count} samples")
        print(f"Failed: {fail_count} samples")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visual Agent Synthesizer")
    parser.add_argument("--file", type=str, required=True, help="Path to config file")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Initialize and run synthesizer
    synthesizer = AgentSynthesizer(args.file, args.workers)
    synthesizer.run_synthesis()


if __name__ == "__main__":
    main()