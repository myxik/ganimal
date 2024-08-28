import torch as th
import wandb
from ganimal.models import Generator
from ganimal.utils import plot_images


def evaluate(config, checkpoint_path):
    # Initialize wandb
    wandb.init(project="gan_project", config=config)

    # Load the generator
    generator = Generator(
        config.training.latent_dim,
        (config.model.channels, config.model.img_size, config.model.img_size),
    )
    generator.load_state_dict(th.load(checkpoint_path))
    generator.eval()

    # Generate images
    with th.no_grad():
        z = th.randn(64, config.training.latent_dim)
        fake_imgs = generator(z)
        fig = plot_images(fake_imgs, 8, 8)
        wandb.log({"evaluated_images": wandb.Image(fig)})


if __name__ == "__main__":
    import yaml

    with open("gan_project/configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    evaluate(config, "path/to/generator_checkpoint.pth")
