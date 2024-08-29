import addict
import yaml
import torch as th
import wandb

from ganimal.models.dcgan import DCGANDiscriminator, DCGANGenerator
from ganimal.datasets.lsun import get_loaders
from ganimal.runners.default_runner import DefaultRunner


def train(config):
    # Initialize wandb
    wandb.init(project="DCGAN LSUN", config=config)

    # Set up dataset, models, optimizers, etc.
    train_loader, valid_loader = get_loaders(config)
    generator = DCGANGenerator(config.latent_dim)
    discriminator = DCGANDiscriminator()

    optimizer_G = th.optim.Adam(
        generator.parameters(),
        lr=config.training.generator.lr,
        betas=(config.training.generator.beta1, 0.999),
    )
    optimizer_D = th.optim.Adam(
        discriminator.parameters(),
        lr=config.training.discriminator.lr,
        betas=(config.training.discriminator.beta1, 0.999),
    )

    runner = DefaultRunner(
        generator,
        discriminator,
        optimizer_G,
        optimizer_D,
        train_loader,
        valid_loader,
        config,
    )
    runner.run()


if __name__ == "__main__":
    with open("./configs/dcgan_lsun.yml", "r") as f:
        config = addict.Dict(yaml.safe_load(f))
    train(config)
