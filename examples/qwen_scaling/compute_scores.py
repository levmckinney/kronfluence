import argparse
import logging
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import default_data_collator

from examples.qwen_scaling.pipeline import (
    construct_model,
    get_openwebtext_dataset,
    get_eval_dataset,
    QWEN_MODELS,
)
from examples.qwen_scaling.task import QwenLanguageModelingTask
from examples.qwen_scaling.save_timing import save_timing_to_csv
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.score_arguments import extreme_reduce_memory_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Default batch sizes by model size
DEFAULT_TRAIN_BATCH_SIZES = {
    "0.5B": 32,
    "1.5B": 16,
    "3B": 8,
    "7B": 4,
    "14B": 2,
    "32B": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Influence score computation for Qwen 2.5 scaling study.")

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
        help="Name of the pre-computed factors.",
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        required=True,
        help="Name of the score.",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples to compute scores over.",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=None,
        help="Number of eval samples to compute scores for.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Batch size for computing query gradients. Defaults based on model size.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=None,
        help="Query batch size",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=True,
        help="Boolean flag to profile computations (default: True).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save timing CSV. If not specified, uses timing_scores_{model_size}.csv",
    )
    args = parser.parse_args()

    # Set default batch size based on model size
    if args.train_batch_size is None:
        args.train_batch_size = DEFAULT_TRAIN_BATCH_SIZES.get(args.model_size, 1)

    # Set default output CSV path
    if args.output_csv is None:
        args.output_csv = f"timing_scores_{args.model_size.lower()}.csv"

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Running score computation for Qwen 2.5 {args.model_size}")

    # Prepare the datasets
    train_dataset = get_openwebtext_dataset(
        model_size=args.model_size,
        num_samples=args.num_train_samples,
    )
    eval_dataset = get_eval_dataset(
        model_size=args.model_size,
        num_samples=args.num_eval_samples,
    )
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Eval dataset size: {len(eval_dataset)}")

    # Prepare the model
    model = construct_model(args.model_size)
    num_layers = model.config.num_hidden_layers
    logging.info(f"Model has {num_layers} layers")

    # Define task and prepare model
    task = QwenLanguageModelingTask(num_layers=num_layers)
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    analyzer = Analyzer(
        analysis_name=f"qwen_scaling_{args.model_size.lower()}",
        model=model,
        task=task,
        profile=args.profile,
    )

    # Configure parameters for DataLoader
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = extreme_reduce_memory_score_arguments(
        damping_factor=None, module_partitions=1, query_gradient_low_rank=rank, dtype=torch.bfloat16
    )
    score_args.query_gradient_accumulation_steps = 1
    score_args.use_full_svd = False
    score_args.precondition_dtype = torch.bfloat16
    score_args.per_sample_gradient_dtype = torch.bfloat16

    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        score_args=score_args,
        factors_name=args.factors_name,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )

    scores = analyzer.load_pairwise_scores(args.scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")

    # Save timing to CSV
    if args.profile:
        save_timing_to_csv(
            profiler=analyzer.profiler,
            output_path=args.output_csv,
            model_size=args.model_size,
            phase="scores",
            extra_metadata={
                "num_train_samples": args.num_train_samples or len(train_dataset),
                "num_eval_samples": args.num_eval_samples or len(eval_dataset),
                "train_batch_size": args.train_batch_size,
            },
        )

    logging.info("Score computation complete.")


if __name__ == "__main__":
    main()
