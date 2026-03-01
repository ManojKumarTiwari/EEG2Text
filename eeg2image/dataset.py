"""BCIC-IV-2a LMDB dataset loader."""

import pickle
import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def to_tensor(array):
    return torch.from_numpy(array).float()


class BCICDataset(Dataset):
    """LMDB-backed BCIC-IV-2a dataset.

    Each sample is a preprocessed EEG trial of shape (22, 4, 200).
    The raw signal is divided by 100 for normalisation.
    """

    def __init__(self, data_dir: str, mode: str = "train"):
        self.db = lmdb.open(
            data_dir, readonly=True, lock=False, readahead=True, meminit=False
        )
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get("__keys__".encode()))[mode]
        self.mode = mode

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        return to_tensor(pair["sample"] / 100), int(pair["label"])


def get_dataloaders(lmdb_dir: str, batch_size: int = 64, num_workers: int = 4):
    """Return (train_loader, val_loader, test_loader, train_set, val_set, test_set)."""
    train_set = BCICDataset(lmdb_dir, "train")
    val_set   = BCICDataset(lmdb_dir, "val")
    test_set  = BCICDataset(lmdb_dir, "test")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_set, val_set, test_set
