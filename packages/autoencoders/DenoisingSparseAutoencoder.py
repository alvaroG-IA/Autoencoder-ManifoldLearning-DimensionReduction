from packages.autoencoders.LinearSparseAutoencoder import LinearSparseAutoencoder
import torch


class DenoisingSparseAutoencoder(LinearSparseAutoencoder):
    def __init__(self, noise_factor: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.noise_factor = noise_factor

    def add_noise(self, data):
        noisy = data + torch.randn_like(data) * self.noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy
