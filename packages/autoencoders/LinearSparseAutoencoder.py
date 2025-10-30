import torch
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder


class LinearSparseAutoencoder(LinearAutoencoder):
    """

    """
    def __init__(self, lambda_l1: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.lambda_l1 = lambda_l1

    def regularization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Redefinición del hook con el fin de añadir regularización L1
        :param embeddings: valores a partir de los que se calcula la regularización
        :return:
        """
        return self.lambda_l1 * torch.sum(torch.abs(embeddings))
