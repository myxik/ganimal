import torch as th
from torchvision import datasets, transforms


def get_loaders(config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),
        ]
    )

    train_dataset = datasets.LSUN(
        root="./data/lsun_data", classes="train", download=True, transform=transform
    )
    test_dataset = datasets.LSUN(
        root="./data/lsun_data", classes="val", download=True, transform=transform
    )

    train_loader = th.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = th.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader
