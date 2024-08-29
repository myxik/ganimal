import torch as th
from torchvision import datasets, transforms


def get_loaders(config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize with mean and std for MNIST
        ]
    )

    train_dataset = datasets.MNIST(
        root="./mnist_data", train=True, download=True, transform=transform
    )
    valid_dataset = datasets.MNIST(
        root="./mnist_data", train=False, download=True, transform=transform
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
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader
