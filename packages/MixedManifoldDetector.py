import numpy as np
import sklearn
from sklearn.manifold import TSNE
from packages.autoencoders.Autoencoder import Autoencoder
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder


class MixedManifoldDetector():
    def __init__(self, input_dim: int = None, autoencoder: Autoencoder = None,
                 manifold_alg: sklearn.base.TransformerMixin = None):

        # SOLUCIÓN TEMPORAL AL PROBLEMA DE INPUT_DIM EN EL AUTOENCODER POR DEFECTO
        if autoencoder is None:
            if input_dim is None:
                raise ValueError("Si no se ha dado un autoencoder en concreto es obligatorio introducir el valor de "
                                 "`input_dim`")
            else:
                self.autoencoder = LinearAutoencoder(input_dim=input_dim)
        else:
            self.autoencoder = autoencoder

        self.manifold_alg = manifold_alg if manifold_alg is not None else TSNE()

    def fit_transform(self, data: np.ndarray):
        pass

    def fit(self, data: np.ndarray):
        pass

    def transform(self, data: np.ndarray):
        pass