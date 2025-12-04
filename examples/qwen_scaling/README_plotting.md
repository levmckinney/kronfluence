# Batch Test Results Plotting

## Overview

The `plot_batch_test_results.py` script visualizes the timing results from the Qwen model scaling experiments conducted with `fit_factors_batch_test.sh`.

## What It Does

The script:
1. **Loads all timing CSV files** from `batch_test_results/` directory
2. **Combines data** into a unified DataFrame for analysis
3. **Creates visualizations** showing how three key phases scale with model size:
   - **Fit Covariance**: Time to fit covariance matrices
   - **Fit Lambda**: Time to fit lambda matrices
   - **Perform Eigendecomposition**: Time to perform eigendecomposition

## Key Features

- **Three-panel plot**: One column for each phase (Fit Covariance, Fit Lambda, Perform Eigendecomposition)
- **Batch size comparison**: Each line represents a different batch size (1, 2, 4, 8, 16, 32)
- **Scaling analysis**: Shows how computation time increases with model size (0.5B → 1.5B → 3B → 7B)
- **Summary statistics**: Prints detailed statistics about timing across all experiments

## Usage

```bash
# Run from the qwen_scaling directory
uv run python plot_batch_test_results.py
```

## Output

- **PDF file**: `timing_by_phase.pdf` - High-quality figure with three subplots
- **Console output**: Summary statistics including:
  - Number of experiments loaded
  - Mean timing by action
  - Mean timing by model size
  - DataFrame information

## Data Structure

The script expects CSV files in the format:
```
model_size,phase,action,mean_duration_s,num_calls,total_duration_s,percentage,num_train_samples,factor_batch_size
0.5B,factors,Perform Eigendecomposition,31.839107,1,31.839107,49.26,500,4
```

## Customization

You can modify the script to:
- Change output format (PDF, PNG, SVG)
- Adjust figure size and layout
- Add log scales for axes
- Filter specific model sizes or batch sizes
- Add additional plot types

## Dependencies

- pandas
- matplotlib
- seaborn
- tueplots
- numpy

All dependencies can be installed via `uv pip install seaborn tueplots`.
