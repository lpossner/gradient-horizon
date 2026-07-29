import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def load_tinystories(
    num_train_stories: int, num_val_stories: int
) -> tuple[list[str], list[str]]:
    dataset = load_dataset("roneneldan/TinyStories")

    train_texts = list(dataset["train"].select(range(num_train_stories))["text"])
    val_texts = list(dataset["validation"].select(range(num_val_stories))["text"])

    return train_texts, val_texts


class GPTDataset(Dataset):
    def __init__(self, ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(ids) - max_length, stride):
            input_chunk = ids[i : i + max_length]
            target_chunk = ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def make_dataloader(
    ids,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):

    dataset = GPTDataset(ids, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
