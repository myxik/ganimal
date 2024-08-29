import torch as th

from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.projection = nn.Linear(latent_dim, 4 * 4 * 1024)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = F.relu(self.projection(x))
        x = rearrange(x, "b (h w c) -> b c h w", h=4, w=4, c=1024)
        return self.layers(x)
    

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.layers(x)
    

if __name__ == "__main__":
    generator = DCGANGenerator(100)
    discriminator = DCGANDiscriminator()

    x = th.randn(1, 100)
    print(generator(x).shape)
    print(discriminator(generator(x)).shape)