<h1 align="center">Reinforced Visual Perception with Tools</h1>

<p align="center">
<strong>Paper | <a href="https://huggingface.co/collections/Frywind/revpt-68b05161d2426128ea5db4d3">Models & Datasets Repo</a></strong>
</p>

## Installation
```bash
conda create -n revpt python=3.10 -y
conda activate revpt
pip install torch==2.6.0 torchvision==0.21.0
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e ".[vllm]"

conda create -n tools python=3.10 -y
conda activate tools
pip install torch==2.4.1 torchvision==0.19.1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install transformers==4.42.0 fastapi uvicorn matplotlib opencv-python python-multipart
```

## Tool services

change config in `tools/tools_config_2.json`
```bash
cd Depth-Anything-V2
mkdir checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

python tools/lanuch_tools.py --config tools_config_2.json
```

## Train
You can download data from [here](https://huggingface.co/datasets/Frywind/REVPT-data). Put them under `data`
generate data using the following command:
```bash
python data/sat_jsonl.py --local-dir [LOCAL_DIR]
```

Change config in `./scripts`
```bash
bash scripts/run.sh
```

## Eval
You can download data from [here](https://huggingface.co/datasets/Frywind/REVPT-data). Put them under `data`
datasets and prompts can be found in `eval/agent_eval.py`
```bash
cd eval
python agent_eval.py --model-name [MODEL_NAME] --port-pool [PORT_POOL] --workers [WORKERS] --dataset [DATASET] --prompt [PROMPT] --evaluate
```
parameter evaluate will use regex to extract answer in \boxed{} to compare with ground truth answer.

or use `benchmark.sh` to run all datasets
