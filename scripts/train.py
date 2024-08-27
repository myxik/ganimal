import torch as th
import wandb
from ganimal.models import Generator, Discriminator
from ganimal.datasets import GANDataset
from ganimal.utils import plot_images

def train(config):
    # Initialize wandb
    wandb.init(project="gan_project", config=config)

    # Set up dataset, models, optimizers, etc.
    dataset = GANDataset(config.data.dataset_path)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    
    generator = Generator(config.training.latent_dim, (config.model.channels, config.model.img_size, config.model.img_size))
    discriminator = Discriminator((config.model.channels, config.model.img_size, config.model.img_size))
    
    optimizer_G = th.optim.Adam(generator.parameters(), lr=config.training.lr, betas=(config.training.beta1, 0.999))
    optimizer_D = th.optim.Adam(discriminator.parameters(), lr=config.training.lr, betas=(config.training.beta1, 0.999))

    # Training loop
    for epoch in range(config.training.epochs):
        for i, real_imgs in enumerate(dataloader):
            # Train discriminator
            # ...

            # Train generator
            # ...

            # Log to wandb
            if i % 100 == 0:
                wandb.log({
                    "D_loss": D_loss.item(),
                    "G_loss": G_loss.item(),
                    "epoch": epoch,
                    "step": i,
                })

        # Generate and log images
        if epoch % 10 == 0:
            with th.no_grad():
                fake_imgs = generator(th.randn(16, config.training.latent_dim))
                fig = plot_images(fake_imgs, 4, 4)
                wandb.log({"generated_images": wandb.Image(fig)})

if __name__ == "__main__":
    import yaml
    with open("gan_project/configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)