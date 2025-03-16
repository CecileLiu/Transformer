import numpy as np

import torch
from torch import nn, Tensor

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Any

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, enc_seq_len: int, dec_seq_len: int, target_seq_len: int, device:str="cuda"):
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len
        self.device = device

        # Precompute valid indices to avoid size mismatch
        self.valid_indices = []
        for idx in range(len(self.data) - self.enc_seq_len - self.target_seq_len - self.dec_seq_len + 1):
            self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        real_idx = self.valid_indices[idx]  # Use precomputed index

        enc_start = real_idx
        enc_end = enc_start + self.enc_seq_len
        dec_end = enc_end + self.dec_seq_len
        target_end = dec_end + self.target_seq_len
        
        encoder_input = self.data[enc_start:enc_end]
        decoder_input = self.data[enc_end:dec_end]
        target = self.data[dec_end:target_end]
        
        return (torch.FloatTensor(encoder_input).to(self.device) ,
            torch.FloatTensor(decoder_input).to(self.device) ,
            torch.FloatTensor(target).to(self.device)) 



