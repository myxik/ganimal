import th
from torch.utils.data import Dataset

class GANDataset(Dataset):
    def __init__(self, data_path):
        self.data = th.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]