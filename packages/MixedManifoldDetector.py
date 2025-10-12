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
                                                                               perplexity=40,
                                                                               n_iter_without_progress=50)

    def fit_transform(self, data: np.ndarray):
        self.autoencoder.fit(data=data, debug=True)
        print("\n\033[92m[AUTOENCODER ENTRENADO CORRECTAMENTE]\033[0m\n")
        embeddings = self.autoencoder.transform(data)
        print("\033[92m[EMBEDDINGS OBTENIDOS CORRECTAMENTE]\033[0m\n")
        print("\033[93m[ENTRENANDO METODO MANIFOLDING ... ]\033[0m\n")
        return self.manifold_alg.fit_transform(embeddings)

    def fit(self, data: np.ndarray):
        self.fit_transform(data)

    def transform(self, data: np.ndarray):
        pass
