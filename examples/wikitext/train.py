import argparse
import logging
import math
import os
import time

import torch
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import default_data_collator

from examples.wikitext.pipeline import apply_fsdp_to_gpt2, construct_gpt2, get_wikitext_dataset
from kronfluence.utils.state import State

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on WikiText dataset.")

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-05,
        help="Fixed learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of epochs to train the model.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1004,
        help="A seed for reproducible training pipeline.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
    )
    parser.add_argument(
        "--apply_fsdp",
        action="store_true",
        help="Whether to apply FSDP to the model.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def train(
    dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    apply_fsdp: bool,
) -> nn.Module:
    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=default_data_collator,
        sampler=torch.utils.data.DistributedSampler(dataset, shuffle=True),
    )

    model = construct_gpt2().to(DEVICE)
    if apply_fsdp:
        model = apply_fsdp_to_gpt2(model)
    else:
        model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_time = time.time()
    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
                labels=batch["labels"].to(device=DEVICE),
            ).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().float() * batch["input_ids"].shape[0]
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")
    return model


def evaluate_model(model: nn.Module, dataset: data.Dataset, batch_size: int) -> float:
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=default_data_collator,
        sampler=torch.utils.data.DistributedSampler(dataset, shuffle=False),
    )

    model.eval()
    total_loss = torch.tensor(0.0, device=DEVICE)
    total_num = torch.tensor(0, device=DEVICE)
    for batch in dataloader:
        with torch.no_grad():
            loss = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
                labels=batch["labels"].to(device=DEVICE),
            ).loss
            total_loss += loss.detach() * batch["input_ids"].shape[0]
            total_num += batch["input_ids"].shape[0]
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_num, op=torch.distributed.ReduceOp.SUM)
    return total_loss.item() / total_num


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    state = State()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = get_wikitext_dataset(split="train")
    model = train(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        apply_fsdp=args.apply_fsdp,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_train_dataset = get_wikitext_dataset(split="eval_train")
    train_loss = evaluate_model(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    train_perplexity = math.exp(train_loss)
    logger.info(f"Train perplexity: {train_perplexity}")

    eval_dataset = get_wikitext_dataset(split="valid")
    eval_loss = evaluate_model(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    try:
        eval_perplexity = math.exp(eval_loss)
    except OverflowError:
        eval_perplexity = float("inf")
    logger.info(f"Evaluation perplexity: {eval_perplexity}")

    if args.apply_fsdp:
        model.unshard()
    else:
        model = model.module

    if args.checkpoint_dir is not None and state.is_main_process:
       torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
