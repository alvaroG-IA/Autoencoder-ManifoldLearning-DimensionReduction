import numpy as np
import sklearn
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from packages.autoencoders.Autoencoder import Autoencoder
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder


class MixedManifoldDetector:
    def __init__(self,
                 autoencoder: Autoencoder = None,
                 manifold_alg: sklearn.base.TransformerMixin = None,
                 seed: int = 42):

        self.seed = seed
        self.autoencoder = autoencoder
        self.manifold_alg = manifold_alg if manifold_alg is not None \
            else TSNE(n_components=2, perplexity=40, n_iter_without_progress=50, random_state=self.seed)

        self.trained = False
        self.train_data = None
        self.train_embeddings = None
        self.train_2d = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """

        :param data: datos con los que se entrenará el sistema y de los que se calculará su proyección 2D
        :return:
        """
        if self.autoencoder is None:
            self.autoencoder = LinearAutoencoder(input_dim=data.shape[1])

        self.autoencoder.fit(data=data, debug=True)
        if self.autoencoder.trained:
            print("\n\033[92m[AUTOENCODER ENTRENADO CORRECTAMENTE]\033[0m\n")
        else:
            raise RuntimeError("El autoencoder no ha sido entrenado correctamente.")

        embeddings = self.autoencoder.transform(data)
        print("\033[92m[EMBEDDINGS OBTENIDOS CORRECTAMENTE]\033[0m\n")

        pts2d = self.manifold_alg.fit_transform(embeddings)
        print("\033[92m[METODO MANIFOLDING ENTRENADO CORRECTAMENTE]\033[0m\n")

        self.train_data = data
        self.train_embeddings = np.array(embeddings)
        self.train_2d = pts2d
        self.trained = True     # marcamos como entrenado correctamente el sistema

        return pts2d

    def fit(self, data: np.ndarray) -> None:
        self.fit_transform(data)

    def transform(self, data: np.ndarray, k: int = 5) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("El sistema debe de primero ser entrenado correctamente.")

        print("\033[93m[EJECUTANDO PROCESO DE INFERENCIA SOBRE TEST]\033[0m\n")

        results = []
        for x in data:
            # Comprobamos si existe un ejemplo igual visto durante el entrenamiento
            mask = np.all(self.train_data == x, axis=1)
            if np.any(mask):
                idx = np.where(mask)[0][0]
                # Almacenamos el resultado 2d obtenido durante el entrenamiento para este ejemplo
                results.append(self.train_2d[idx])
            else:
                embedding = self.autoencoder.transform(x.reshape(1, -1))
                # Calculamos el vector de distancias entre el nuevo embedding y los vistos en el entrenamiento
                distances = pairwise_distances(embedding, self.train_embeddings, metric='euclidean')[0]
                # Índices de los k vecinos más cercanos al embedding nuevo
                idxs = np.argsort(distances)[:k]

                vecinos2d = self.train_2d[idxs]
                avg_2dpoint = np.mean(vecinos2d, axis=0).reshape(1, -1)
                results.append(avg_2dpoint)

        return np.vstack(results)

    def evaluate(self, n_neighbors: int = 5, data_fraction: float = 1.0) -> float:
        if not (0 <= data_fraction <= 1):
            raise ValueError("El parámetro 'data_fraction' debe estar en el rango [0, 1].")

        if not self.trained:
            raise RuntimeError("El modelo debe estar entrenado antes de ejecutar 'evaluate'.")

        split = int(data_fraction * len(self.train_data))
        print(f"\033[94m[TRUSTWORTHINESS RESULTS]\033[0m")

        tw_final = trustworthiness(
            self.train_data[:split],
            self.train_2d[:split],
            n_neighbors=n_neighbors,
        )

        print(f" - Trustworthiness de los primeros {split} valores (Original -> 2D): {tw_final:.4f}")
        return tw_final
