# Qwen 2.5 Scaling Example

This example demonstrates influence computation across the Qwen 2.5 model family for scaling studies. It supports models from 0.5B to 32B parameters and saves timing data to CSV files.

## Supported Models

| Model Size | HuggingFace ID | Layers | Recommended FSDP |
|------------|----------------|--------|------------------|
| 0.5B | Qwen/Qwen2.5-0.5B | 24 | No |
| 1.5B | Qwen/Qwen2.5-1.5B | 28 | No |
| 3B | Qwen/Qwen2.5-3B | 36 | No |
| 7B | Qwen/Qwen2.5-7B | 28 | Optional |
| 14B | Qwen/Qwen2.5-14B | 48 | Yes |
| 32B | Qwen/Qwen2.5-32B | 64 | Yes |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Fit Influence Factors

```bash
python -m examples.qwen_scaling.fit_factors \
    --model_size 0.5B \
    --factors_name qwen_0.5b_factors \
    --num_train_samples 1000 \
    --output_csv timing_factors_0.5b.csv
```

**Arguments:**
- `--model_size` (required): One of 0.5B, 1.5B, 3B, 7B, 14B, 32B
- `--factors_name` (required): Name for saving computed factors
- `--num_train_samples`: Number of training samples for Hessian fitting (default: all)
- `--factor_batch_size`: Batch size (auto-selected based on model size)
- `--use_fsdp`: Enable FSDP for larger models
- `--output_csv`: Path to save timing CSV

### Step 2: Compute Influence Scores

```bash
python -m examples.qwen_scaling.compute_scores \
    --model_size 0.5B \
    --factors_name qwen_0.5b_factors \
    --scores_name qwen_0.5b_scores \
    --num_train_samples 1000 \
    --num_eval_samples 100 \
    --output_csv timing_scores_0.5b.csv
```

**Arguments:**
- `--model_size` (required): Model size (must match factors)
- `--factors_name` (required): Name of pre-computed factors
- `--scores_name` (required): Name for saving scores
- `--num_train_samples`: Training samples for scoring
- `--num_eval_samples`: Eval samples for scoring
- `--train_batch_size`: Batch size (auto-selected based on model size)
- `--output_csv`: Path to save timing CSV

## Examples for Different Scales

### Small Model (0.5B)
```bash
python -m examples.qwen_scaling.fit_factors \
    --model_size 0.5B \
    --factors_name qwen_0.5b \
    --num_train_samples 1000

python -m examples.qwen_scaling.compute_scores \
    --model_size 0.5B \
    --factors_name qwen_0.5b \
    --scores_name qwen_0.5b_scores \
    --num_train_samples 1000 \
    --num_eval_samples 100
```

### Medium Model (7B)
```bash
python -m examples.qwen_scaling.fit_factors \
    --model_size 7B \
    --factors_name qwen_7b \
    --num_train_samples 500 \
    --factor_batch_size 2

python -m examples.qwen_scaling.compute_scores \
    --model_size 7B \
    --factors_name qwen_7b \
    --scores_name qwen_7b_scores \
    --num_train_samples 500 \
    --num_eval_samples 50
```

### Large Model (32B) with FSDP
```bash
python -m examples.qwen_scaling.fit_factors \
    --model_size 32B \
    --factors_name qwen_32b \
    --num_train_samples 200 \
    --use_fsdp

python -m examples.qwen_scaling.compute_scores \
    --model_size 32B \
    --factors_name qwen_32b \
    --scores_name qwen_32b_scores \
    --num_train_samples 200 \
    --num_eval_samples 20
```

## CSV Output Format

The timing CSV files have the following columns:

| Column | Description |
|--------|-------------|
| model_size | Model size (e.g., "0.5B") |
| phase | "factors" or "scores" |
| action | Operation name (e.g., "Fit Covariance") |
| mean_duration_s | Mean duration in seconds |
| num_calls | Number of times the action was called |
| total_duration_s | Total duration in seconds |
| percentage | Percentage of total time |
| num_train_samples | Number of training samples used |
| num_eval_samples | Number of eval samples (scores only) |
| factor_batch_size | Batch size used (factors only) |
| train_batch_size | Batch size used (scores only) |

**Example CSV:**
```csv
model_size,phase,action,mean_duration_s,num_calls,total_duration_s,percentage,num_train_samples,factor_batch_size
0.5B,factors,Fit Covariance,12.345678,1,12.345678,45.2,1000,16
0.5B,factors,Perform Eigendecomposition,8.234567,1,8.234567,30.1,1000,16
0.5B,factors,Fit Lambda,6.789012,1,6.789012,24.7,1000,16
```

## Default Batch Sizes

The scripts automatically select batch sizes based on model size:

| Model Size | Factor Batch Size | Train Batch Size |
|------------|-------------------|------------------|
| 0.5B | 16 | 32 |
| 1.5B | 8 | 16 |
| 3B | 4 | 8 |
| 7B | 2 | 4 |
| 14B | 1 | 2 |
| 32B | 1 | 1 |

You can override these with `--factor_batch_size` and `--train_batch_size`.

## Multi-GPU Training

Both scripts work with `torchrun` and `accelerate launch` for distributed training.

### Using torchrun

```bash
# Fit factors on 4 GPUs
torchrun --nproc_per_node=4 -m examples.qwen_scaling.fit_factors \
    --model_size 7B \
    --factors_name qwen_7b_factors \
    --num_train_samples 1000 \
    --output_csv timing_factors_7b.csv

# Compute scores on 4 GPUs
torchrun --nproc_per_node=4 -m examples.qwen_scaling.compute_scores \
    --model_size 7B \
    --factors_name qwen_7b_factors \
    --scores_name qwen_7b_scores \
    --num_train_samples 1000 \
    --num_eval_samples 100 \
    --output_csv timing_scores_7b.csv
```

### Using accelerate launch

```bash
# Fit factors on 4 GPUs
accelerate launch --num_processes=4 -m examples.qwen_scaling.fit_factors \
    --model_size 7B \
    --factors_name qwen_7b_factors \
    --num_train_samples 1000 \
    --output_csv timing_factors_7b.csv

# Compute scores on 4 GPUs
accelerate launch --num_processes=4 -m examples.qwen_scaling.compute_scores \
    --model_size 7B \
    --factors_name qwen_7b_factors \
    --scores_name qwen_7b_scores \
    --num_train_samples 1000 \
    --num_eval_samples 100 \
    --output_csv timing_scores_7b.csv
```

### Multi-node Training

For multi-node setups with torchrun:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m examples.qwen_scaling.fit_factors \
    --model_size 32B \
    --factors_name qwen_32b_factors \
    --num_train_samples 500 \
    --use_fsdp
```

## Hardware Requirements

- **0.5B-3B**: Single GPU with 24GB+ VRAM
- **7B**: Single GPU with 40GB+ VRAM or multiple GPUs
- **14B-32B**: Multiple GPUs with FSDP recommended
