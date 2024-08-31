import torch as th

from ganimal.runners.default_runner import DefaultRunner


class WGANRunner(DefaultRunner):
    def __init__(self, generator, discriminator, optimizer_G, optimizer_D, trainloader, validloader, config):
        super().__init__(generator, discriminator, optimizer_G, optimizer_D, trainloader, validloader, config)

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

            for p in self.discriminator.parameters():
                p.data.clamp_(-self.config.clip_value, self.config.clip_value)

            D_loss_mean += D_loss.item()
        
        return D_loss_mean / self.config.k

    def _compute_discriminator_loss(self, D_real, D_fake):
        return -(th.mean(D_real) - th.mean(D_fake))
    
    def _compute_generator_loss(self, D_fake):
        return -th.mean(D_fake)