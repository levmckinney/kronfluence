import argparse
import logging
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import default_data_collator

from examples.qwen_scaling.pipeline import construct_model, get_openwebtext_dataset, QWEN_MODELS, apply_fsdp_to_model
from examples.qwen_scaling.task import QwenLanguageModelingTask
from examples.qwen_scaling.save_timing import save_timing_to_csv
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import smart_low_precision_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Default batch sizes by model size
DEFAULT_FACTOR_BATCH_SIZES = {
    "0.5B": 16,
    "1.5B": 8,
    "3B": 4,
    "7B": 2,
    "14B": 1,
    "32B": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Influence factor computation for Qwen 2.5 scaling study.")

    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=list(QWEN_MODELS.keys()),
        help="Model size to use.",
    )
    parser.add_argument(
        "--factors_name",
        type=str,
        required=True,
        help="Name of the factor.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=None,
        help="Batch size for computing influence factors. Defaults based on model size.",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples to use for Hessian fitting.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=True,
        help="Boolean flag to profile computations (default: True).",
    )
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        default=False,
        help="Boolean flag to use FSDP.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save timing CSV. If not specified, uses timing_factors_{model_size}.csv",
    )
    args = parser.parse_args()

    # Set default batch size based on model size
    if args.factor_batch_size is None:
        args.factor_batch_size = DEFAULT_FACTOR_BATCH_SIZES.get(args.model_size, 1)

    # Set default output CSV path
    if args.output_csv is None:
        args.output_csv = f"timing_factors_{args.model_size.lower()}.csv"

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    # Prepare the model
    model = construct_model(args.model_size)
    num_layers = model.config.num_hidden_layers
    logging.info(f"Model has {num_layers} layers")

    # Define task and prepare model
    task = QwenLanguageModelingTask(num_layers=num_layers)

    model = prepare_model(model, task)
    model = apply_fsdp_to_model(model)

    logging.info(f"Running factor computation for Qwen 2.5 {args.model_size}")

    # Prepare the dataset
    train_dataset = get_openwebtext_dataset(
        model_size=args.model_size,
        num_samples=args.num_train_samples,
    )
    logging.info(f"Training dataset size: {len(train_dataset)}")

    analyzer = Analyzer(
        analysis_name=f"qwen_scaling_{args.model_size.lower()}",
        model=model,
        task=task,
        profile=args.profile,
    )

    # Configure parameters for DataLoader
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    factors_name = args.factors_name
    factor_args = smart_low_precision_factor_arguments(
        strategy=args.factor_strategy,
        dtype=torch.bfloat16,
    )
    factor_args.covariance_module_partitions = 1
    factor_args.lambda_module_partitions = 1
    factor_args.covariance_data_partitions = 1
    factor_args.lambda_data_partitions = 1

    factor_args.shard_covariance = args.use_fsdp
    factor_args.shard_lambda = args.use_fsdp
    factor_args.shard_eigendecomposition = args.use_fsdp

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )

    # Save timing to CSV
    if args.profile:
        save_timing_to_csv(
            profiler=analyzer.profiler,
            output_path=args.output_csv,
            model_size=args.model_size,
            phase="factors",
            extra_metadata={
                "num_train_samples": args.num_train_samples or len(train_dataset),
                "factor_batch_size": args.factor_batch_size,
            },
        )

    logging.info("Factor computation complete.")


if __name__ == "__main__":
    main()
