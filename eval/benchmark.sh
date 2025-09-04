#!/bin/bash

# Set default parameters
MODEL_NAME=${MODEL_NAME:-"Qwen2.5-VL-7B-0814"}
PORT_POOL=${PORT_POOL:-"10014,10015,10016,10017"}
WORKERS=${WORKERS:-32}
PROMPT=${PROMPT:-"agent"}
REMOTE=${REMOTE:-false}
EVALUATE=${EVALUATE:-true}

# Supported dataset list
DATASETS=(
    "cvbench"
    "mmstar"
    "blink"
    "mmmu"
    "mmbench"
    "mathvista"
    "blink-hard"
)

# Show usage instructions
show_usage() {
    echo "Usage: $0 [OPTIONS] [DATASET1 DATASET2 ...]"
    echo ""
    echo "Options:"
    echo "  --model-name MODEL     Model name (default: Qwen2.5-VL)"
    echo "  --port-pool PORTS      API server port pool, comma separated (default: 8015,8016,8017,8018)"
    echo "  --workers NUM          Number of parallel worker threads (default: 32)"
    echo "  --prompt TYPE          Prompt type (default: agent)"
    echo "  --remote               Use remote API (default: false)"
    echo "  --evaluate             Enable answer verification (default: false)"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Available datasets:"
    for dataset in "${DATASETS[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all datasets with default settings"
    echo "  $0 cvbench mmstar                     # Run only cvbench and mmstar"
    echo "  $0 --workers 16 --evaluate cvbench    # Run cvbench with 16 workers and evaluation enabled"
    echo "  $0 --remote --model-name gpt-4o       # Run all datasets with remote API and gpt-4o model"
}

# Parse command line arguments
SELECTED_DATASETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port-pool)
            PORT_POOL="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --remote)
            REMOTE=true
            shift
            ;;
        --evaluate)
            EVALUATE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Check if it's a valid dataset name
            if [[ " ${DATASETS[@]} " =~ " ${1} " ]]; then
                SELECTED_DATASETS+=("$1")
            else
                echo "Unknown dataset: $1"
                echo "Available datasets: ${DATASETS[*]}"
                exit 1
            fi
            shift
            ;;
    esac
done

# If no dataset is specified, use all datasets
if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
    SELECTED_DATASETS=("${DATASETS[@]}")
fi

# Build base command arguments
BASE_ARGS=(
    "--model-name" "$MODEL_NAME"
    "--port-pool" "$PORT_POOL"
    "--workers" "$WORKERS"
    "--prompt" "$PROMPT"
)

if [[ "$REMOTE" == "true" ]]; then
    BASE_ARGS+=("--remote")
fi

if [[ "$EVALUATE" == "true" ]]; then
    BASE_ARGS+=("--evaluate")
fi

# Show configuration information
echo "=========================================="
echo "Dataset Evaluation Configuration"
echo "=========================================="
echo "Model Name: $MODEL_NAME"
echo "Port Pool: $PORT_POOL"
echo "Workers: $WORKERS"
echo "Prompt Type: $PROMPT"
echo "Remote API: $REMOTE"
echo "Evaluation: $EVALUATE"
echo "Selected Datasets: ${SELECTED_DATASETS[*]}"
echo "=========================================="
echo ""

# Create log directory
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Run evaluation
TOTAL_DATASETS=${#SELECTED_DATASETS[@]}
CURRENT=0
FAILED_DATASETS=()

for dataset in "${SELECTED_DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_DATASETS] Starting evaluation for dataset: $dataset"
    echo "Time: $(date)"
    
    # Set log file
    LOG_FILE="$LOG_DIR/${dataset}.log"
    
    # Run evaluation command
    python agent_eval.py "${BASE_ARGS[@]}" --dataset "$dataset" 2>&1 | tee "$LOG_FILE"
    
    # Check if successful
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "✅ Dataset $dataset completed successfully"
    else
        echo "❌ Dataset $dataset failed"
        FAILED_DATASETS+=("$dataset")
    fi
    
    echo "Log saved to: $LOG_FILE"
    echo "----------------------------------------"
    echo ""
done

# Show summary
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Total datasets: $TOTAL_DATASETS"
echo "Successful: $((TOTAL_DATASETS - ${#FAILED_DATASETS[@]}))"
echo "Failed: ${#FAILED_DATASETS[@]}"

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    echo "Failed datasets: ${FAILED_DATASETS[*]}"
fi

echo "All logs saved in: $LOG_DIR"
echo "Evaluation completed at: $(date)"
echo "=========================================="

# If there are failed datasets, exit code is 1
if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    exit 1
fi