import torch as th
from torchvision import transforms
from datasets import load_dataset


class LSUN(th.utils.data.Dataset):
    def __init__(self, split, transforms):
        self.dataset = load_dataset("pcuenq/lsun-bedrooms", split=split)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.transforms:
            return self.transforms(self.dataset[idx]["image"])
        else:
            return self.dataset[idx]


def get_loaders(config):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),
        ]
    )

    train_dataset = LSUN(split="train", transforms=transform)
    valid_dataset = LSUN(split="test", transforms=transform)

    train_loader = th.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = th.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader
