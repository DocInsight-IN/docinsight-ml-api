
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class SiameseDataset(Dataset):
    def __init__(self, pairs: List[Tuple], labels: List[str], tokenizer, max_length: int):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        encoded_pair = self.tokenizer(pair[0], pair[1], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return encoded_pair, label