import torch
from sklearn.exceptions import NotFittedError
from torch.utils.data import DataLoader
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder
import numpy as np
import torch.nn as nn


class LinearSparseAutoencoder(LinearAutoencoder):
    def __init__(self, lambda_l1: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.lambda_l1 = lambda_l1

    def regularization(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.lambda_l1 * torch.sum(torch.abs(embeddings))
