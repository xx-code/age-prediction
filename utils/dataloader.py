import torch
from torch.utils.data import Dataset

class FaceDataset(Dataset):

    def __init__(self):
        super().__init__()
    
    def __len__(self):
        return 0
    
    def __getitem__(self, index) :
        return super().__getitem__(index)