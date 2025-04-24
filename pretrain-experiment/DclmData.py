# Adapted from https://github.com/Lightning-AI/litgpt/blob/f6031e3a88e272ec86ad8f412573699589f4d41b/litgpt/data/lit_data.py
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from litgpt.data import DataModule


@dataclass
class DclmData(DataModule):
    """Use litdata's CombinedStreamingDataset to load the DCLM-Baseline-1.0 dataset as we have stored and tokenized in on the cluster."""

    def __init__(self, data_path=None, seed=42, num_workers=8):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = 1  
        self.seq_length = 2048

        # find the number of data parts
        self.num_parts = 0
        while os.path.exists(os.path.join(self.data_path, f"part{self.num_parts+1}-train")):
            self.num_parts += 1

        if self.num_parts < 1:
            raise ValueError("No data parts found")

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def train_dataloader(self) -> DataLoader:
        from litdata import StreamingDataset, StreamingDataLoader, TokensLoader, CombinedStreamingDataset

        datasets = [
            StreamingDataset(
                input_dir=os.path.join(self.data_path, f"part{part_idx+1}-train"), 
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
                seed=self.seed,
            ) for part_idx in range(self.num_parts)
        ]
        dataset = CombinedStreamingDataset(datasets=datasets, seed=self.seed)
        dataloader = StreamingDataLoader(
            dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=3, drop_last=True
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """We use the validation data of the first data part."""
        from litdata import StreamingDataset, StreamingDataLoader, TokensLoader, CombinedStreamingDataset

        datasets = [
            StreamingDataset(
                input_dir=os.path.join(self.data_path, f"part1-val"), 
                item_loader=TokensLoader(block_size=self.seq_length),
                subsample=0.3,        # ca. 100M tokens for validation
                shuffle=True,
                drop_last=True,
                seed=7,               # fix the order of the validation data
            ) 
        ]
        dataset = CombinedStreamingDataset(datasets=datasets, seed=self.seed)
        dataloader = StreamingDataLoader(
            dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=3, drop_last=True
        )
        return dataloader
