from omegaconf import DictConfig

# Import Agent class and tool prompts from verl
from verl.api_agent import Agent

# Create Agent configuration
agent_config = DictConfig({
    "max_turns": 5,
    "max_tokens_per_turn": 2048,
    "save_intermediate_responses": True,
    "openai_api_key": "",
    "openai_model": "gpt-4.1",
    "openai_temperature": 1.0,
})

# Initialize Agent
agent = Agent(agent_config)

import json
import argparse

parser = argparse.ArgumentParser(description="Visual Agent")
parser.add_argument("--file", type=str, required=True, help="Path to config file")
args = parser.parse_args()

with open(args.file, 'r') as f:
    ds = json.load(f)
    
# Set result save path and image save directory
import os
import json
import time
import base64
from tqdm import tqdm
import io
from PIL import Image
import concurrent.futures
import threading
# from verl.prompt import get_system_prompt
from prompt_g import get_system_prompt

timestamp = time.strftime('%Y%m%d_%H%M%S')
output_path = f"syn/fix_dep.jsonl"
image_dir = f"syn/fix_dep"
os.makedirs(image_dir, exist_ok=True)

# Ensure output file exists
if not os.path.exists(output_path):
    with open(output_path, 'w') as f:
        pass

# Create thread lock for synchronized file writing
file_lock = threading.Lock()

# Process base64 image and save to file
def save_base64_image(base64_str, image_dir, sample_id, image_index):
    # Extract base64 string from data URL
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    
    # Decode base64 string
    image_data = base64.b64decode(base64_str)
    
    # Determine image format
    image_format = "png"  # Default format
    
    # Create filename
    filename = f"{sample_id}_{image_index}.{image_format}"
    filepath = os.path.join(image_dir, filename)
    
    # Save image
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    return filename

# Process images in conversation
def process_conversation_images(conversation, image_dir, sample_id):
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
                        # Save base64 image
                        filename = save_base64_image(url, image_dir, sample_id, image_index)
                        image_index += 1
                        # Replace with relative path
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
    
# Function to process single sample
def process_sample_sat(args):
    idx, sample, agent, image_dir = args
    try:
        sample = sample['metadata']
        # Get sample data
        sample_image = Image.open(sample['images'][0])
        qa = sample['processed_mcqa']
        sample_prompt = qa['question']
        options = qa['options']
        
        if qa['options']:
            answer = qa['options'][qa['answer']]
            sample_prompt += f"\nAnswer from the following choices:"
            for i, option in enumerate(options):
                sample_prompt += f"\n{chr(65 + i)}. {option}"
            sample_prompt += '\nPut answer letter in the \\boxed{}'
        else:
            answer = qa['answer']
            
        # Call Agent for conversation
        response, conversation = agent.chat_with_tools(
            system_prompt=get_system_prompt(),
            prompt='<image>' + sample_prompt, 
            images=[sample_image]
        )
        
        # Process and save images in conversation
        processed_conversation = process_conversation_images(conversation, image_dir, idx)
        
        # Prepare result
        result = {
            'sample_id': idx,
            'question': qa['question'],
            'answer': chr(65 + options.index(answer)) if options else answer,  # Convert answer index to letter
            'response': response,
            'conversation': processed_conversation,
            'metadata': sample
        }
        
        # Use lock to ensure thread-safe file writing
        with file_lock:
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
        
        return idx, True
    except Exception as e:
        print(f"Error processing sample {idx}: {str(e)}")
        return idx, False

# Run evaluation and save results
test_dataset = ds

# Prepare argument list
args_list = [(idx, sample, agent, image_dir) 
             for idx, sample in enumerate(test_dataset)]

import random
random.seed(42)

random.shuffle(args_list)

max_workers = 16  # Set number of parallel worker threads

print(f"Starting evaluation of {len(args_list)} samples using {max_workers} parallel worker threads")

# Use thread pool for parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(process_sample_sat, args_list), total=len(args_list)))

# Count successful and failed samples
success_count = sum(1 for _, success in results if success)
fail_count = len(results) - success_count

print(f"Evaluation complete, results saved to {output_path}")
print(f"Successfully processed: {success_count} samples, Failed: {fail_count} samples")