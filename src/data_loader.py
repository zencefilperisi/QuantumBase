# src/data_loader.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class GenomicSignalDataset(Dataset):
    """
    Handles real-world genomic signals. 
    Implements Median-MAD normalization which is the gold standard for Nanopore data.
    """
    def __init__(self, signals, labels, seq_length=30):
        self.signals = signals
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        # Robust Scaling (Median Absolute Deviation)
        # Real sensors have drift; this centers them at 0
        median = np.median(signal)
        mad = np.median(np.abs(signal - median)) + 1e-9
        norm_signal = (signal - median) / mad
        
        return torch.tensor(norm_signal, dtype=torch.float32).unsqueeze(-1), torch.tensor(label, dtype=torch.long)