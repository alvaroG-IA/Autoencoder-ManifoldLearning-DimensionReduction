import numpy as np
import sklearn
from sklearn.manifold import TSNE
from packages.autoencoders.Autoencoder import Autoencoder
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder


class MixedManifoldDetector:
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

        self.manifold_alg = manifold_alg if manifold_alg is not None else TSNE(n_components=2,
                                                                               perplexity=20,
                                                                               n_iter_without_progress=25)

    def fit_transform(self, data: np.ndarray):
        self.autoencoder.fit(data=data, debug=True)
        embeddings = self.autoencoder.transform(data)
        return self.manifold_alg.fit_transform(embeddings)

    def fit(self, data: np.ndarray):
        self.fit_transform(data)

    def transform(self, data: np.ndarray):
        pass
