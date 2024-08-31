import torch as th
import wandb

from torch.nn import functional as F
from torchvision.utils import make_grid


class DefaultRunner:
    def __init__(
        self,
        generator,
        discriminator,
        optimizer_G,
        optimizer_D,
        trainloader,
        validloader,
        config,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.trainloader = trainloader
        self.validloader = validloader
        self.config = config
        self.global_step = 0
        self.device = th.device(config.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def run(self):
        self.train()

    def train(self):
        self.generator.train()
        self.discriminator.train()
        for e in range(self.config.num_epochs):
            D_loss_mean_epoch, G_loss_epoch = self._train_epoch()
            self._log_epoch(e, D_loss_mean_epoch, G_loss_epoch)
            self.evaluate(e)

    def _train_epoch(self):
        D_loss_mean_epoch = G_loss_epoch = 0
        for batch in self.trainloader:
            real_images = batch.to(self.device)
            batch_size = real_images.size(0)

            D_loss_mean = self._train_discriminator(real_images, batch_size)
            G_loss = self._train_generator(batch_size)

            G_loss_epoch += G_loss.item()
            D_loss_mean_epoch += D_loss_mean

            self._log_step(D_loss_mean, G_loss)

        return D_loss_mean_epoch / len(self.trainloader), G_loss_epoch / len(self.trainloader)

    def _train_discriminator(self, real_images, batch_size):
        D_loss_mean = 0
        for _ in range(self.config.k):
            z = th.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_images = self.generator(z).detach()

            D_real = self.discriminator(real_images)
            D_fake = self.discriminator(fake_images)

            D_loss = self._compute_discriminator_loss(D_real, D_fake)

            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()

            D_loss_mean += D_loss.item()
        
        return D_loss_mean / self.config.k

    def _train_generator(self, batch_size):
        z = th.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_images = self.generator(z)
        D_fake = self.discriminator(fake_images)

        G_loss = self._compute_generator_loss(D_fake)

        self.optimizer_G.zero_grad()
        G_loss.backward()
        self.optimizer_G.step()

        return G_loss

    def _compute_discriminator_loss(self, D_real, D_fake):
        real_loss = F.binary_cross_entropy_with_logits(
            D_real, th.ones_like(D_real)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            D_fake, th.zeros_like(D_fake)
        )
        return real_loss + fake_loss
    
    def _compute_generator_loss(self, D_fake):
        return F.binary_cross_entropy_with_logits(D_fake, th.ones_like(D_fake))

    def _log_step(self, D_loss_mean, G_loss):
        wandb.log(
            {
                "D_loss_mean": D_loss_mean,
                "G_loss": G_loss,
            }
        )

    def _log_epoch(self, epoch, D_loss_mean_epoch, G_loss_epoch):
        wandb.log(
            {
                "epoch": epoch,
                "D_loss_mean_epoch": D_loss_mean_epoch / len(self.trainloader),
                "G_loss_epoch": G_loss_epoch / len(self.trainloader),
            }
        )

    def evaluate(self, epoch):
        self.generator.eval()
        self.discriminator.eval()

        with th.no_grad():
            D_loss_mean = 0
            G_loss_mean = 0
            for batch in self.validloader:
                real_images = batch
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                z = th.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_images = self.generator(z)
                D_fake = self.discriminator(fake_images)

                D_real = self.discriminator(real_images)

                real_loss = F.binary_cross_entropy_with_logits(
                    D_real, th.ones_like(D_real)
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    D_fake, th.zeros_like(D_fake)
                )

                D_loss = real_loss + fake_loss
                D_loss_mean += D_loss.item()

                G_loss = F.binary_cross_entropy_with_logits(
                    D_fake, th.ones_like(D_fake)
                )
                G_loss_mean += G_loss.item()

            self.generate_images(epoch)
            self.global_step += 1

            # Log to wandb
            wandb.log(
                {
                    "valid_D_loss": D_loss_mean / len(self.validloader),
                    "valid_G_loss": G_loss_mean / len(self.validloader),
                    "epoch": epoch,
                }
            )

    def generate(self):
        pass

    def generate_images(self, epoch):
        if epoch % self.config.save_image_epochs == 0:
            self.generator.eval()
            with th.no_grad():
                z = th.randn(
                    self.config.save_image_batch_size,
                    self.config.latent_dim,
                    device=self.device,
                )
                fake_images = self.generator(z)

                grid = make_grid(fake_images, nrow=5, normalize=True, scale_each=True)
                grid = F.interpolate(grid, scale_factor=2, mode="nearest")
                wandb.log(
                    {
                        "fake_images": wandb.Image(grid),
                        "epoch": epoch,
                    }
                )
