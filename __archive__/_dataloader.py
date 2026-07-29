"""
Dataloaders for pretraining (simplified for single node).
"""

import torch
import pyarrow.parquet as pq

from minichat.dataset import list_parquet_files
from minichat.common import print0


def _document_batches(split, tokenizer_batch_size):
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, (
        "No dataset parquet files found, did you run dataset.py?"
    )
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    # Infinite iterator
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size]


def tokenizing_data_loader(
    tokenizer,
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cpu",
    buffer_size=1000,
):
    """
    Simpler version of the BEST-FIT dataloader, single process.
    """
    row_capacity = T + 1
    batches = _document_batches(split, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []

    def refill_buffer():
        doc_batch = next(batches)
        token_lists = tokenizer.encode(
            doc_batch, prepend=bos_token, num_threads=tokenizer_threads
        )
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers
    use_device = device != "cpu"  # pin memory if using a device
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_device)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Best fit strategy
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos : pos + doc_len] = torch.tensor(
                        doc, dtype=torch.long
                    )
                    pos += doc_len
                else:
                    # Crop shortest
                    shortest_idx = min(
                        range(len(doc_buffer)), key=lambda i: len(doc_buffer[i])
                    )
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        gpu_buffer.copy_(cpu_buffer)
        yield inputs, targets
