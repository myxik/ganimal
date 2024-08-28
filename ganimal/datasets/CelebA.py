import cv2
import pandas as pd
import torch as th

from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CelebADataset(Dataset):
    def __init__(self, data_path, data_frame_path, split, transform=None):
        self.data_path = Path(data_path)
        df = pd.read_csv(data_frame_path)
        if split == 'train':
            self.df = df[df.partition == 0]
        elif split == 'val':
            self.df = df[df.partition == 1]
        elif split == 'test':
            self.df = df[df.partition == 2]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        image = cv2.imread(str(self.data_path / img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        image = cv2.resize(image, (224, 224))
        
        if self.transform:
            image = self.transform(image)

        return image


def get_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CelebADataset("./data/celeba_data/img_align_celeba/img_align_celeba", "./data/celeba_data/list_eval_partition.csv", "train", transform=transform)
    valid_dataset = CelebADataset("./data/celeba_data/img_align_celeba/img_align_celeba", "./data/celeba_data/list_eval_partition.csv", "val", transform=transform)

    train_loader = th.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True
    )
    valid_loader = th.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=32, pin_memory=True, drop_last=False
    )

    return train_loader, valid_loader

