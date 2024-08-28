import torch as th

import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.unfold = nn.Linear(latent_dim, 7 * 7 * 64)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.Tanh(),
        )

    def forward(self, x):
        x = F.relu(self.unfold(x))
        x = rearrange(x, "b (h w c) -> b c h w", h=7, w=7, c=64)
        return self.layers(x)
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1),  # Output to 1
        )

    def forward(self, x):
        return self.layers(x)
    

class CGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.unfold = nn.Linear(latent_dim, 7 * 7 * 64)

        self.layers = nn.Sequential(
        nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28x28
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 56x56
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 112x112
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # 224x224
        nn.Tanh(),
        )

    def forward(self, x):
        x = F.relu(self.unfold(x))
        x = rearrange(x, "b (h w c) -> b c h w", h=7, w=7, c=64)
        return self.layers(x)
    

class CDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 7 x 7
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.layers(x)
    