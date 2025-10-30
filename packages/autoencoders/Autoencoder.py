import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from sklearn.exceptions import NotFittedError


class Autoencoder(nn.Module, ABC):
    """
    Clase base abstracta para distintos tipos de autoencoders.
    Contiene la lógica común de entrenamiento (fit) y transformación (transform).
    Las subclases solo necesitan definir el forward (encoder+decoder) y encoce (encoder)
    y, opcionalmente, un término de regularización y ruido.
    """

    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 32,
                 epochs: int = 100,
                 loss_threshold: float = 0,
                 min_delta: float = 1e-6,
                 max_num_iters_without_progress: int = 20,
                 batch_size: int = 32,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 lr: float = 1e-3,
                 loss_fn: torch.nn.Module = nn.MSELoss(),
                 data_scaled: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.loss_threshold = loss_threshold
        self.min_delta = min_delta
        self.max_num_iters_without_progress = max_num_iters_without_progress
        self.batch_size = batch_size
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.loss_fn = loss_fn
        self.data_scaled = data_scaled
        self.trained = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Método abstracto de ejecución del modelo. Debe devolver (embedding, reconstrucción)
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Método abstracto de ejecución del encoder del modelo. Debe devolver (embedding)
        """
        pass

    def regularization(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Hook para añadir regularización.
        Por defecto no añade nada.
        """
        return torch.tensor(0.0)

    def add_noise(self, data) -> torch.Tensor:
        """
        Hook para añadir ruido a los datos de entrada.
        Por defecto no añadirá ningún ruido.
        """
        return data

    def fit(self, data: np.ndarray, debug: bool = False):
        """
        Método de entrenamiento genérico.
        :param data: datos de entrenamiento
        :param debug: imprimir logs
        """
        # Convertimos el formato de los datos a uno a coder al modelo
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)

        self.train()

        last_avg_loss = np.inf
        counter = 0

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                noisy_batch = self.add_noise(batch)
                embedding, recon = self(noisy_batch)
                loss = self.loss_fn(recon, batch)
                loss += self.regularization(embedding)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # Early stop por perdida menor a cota seleccionada
            if avg_loss < self.loss_threshold:
                print(f"Entrenamiento detenido en la época {epoch+1}: pérdida {avg_loss:.6f} "
                      f"< threshold {self.loss_threshold:.6f}")
                break

            # Early stop por falta de mejora en número de iteraciones
            if (last_avg_loss - avg_loss) < self.min_delta:
                counter += 1
                if counter == self.max_num_iters_without_progress:
                    print(f"Entrenamiento detenido en la época {epoch+1} por falta de mejora ({self.min_delta} "
                          f"durante las últmas {self.max_num_iters_without_progress} épocas). Última pérdida: {avg_loss:.6f}")
                    break
            else:
                counter = 0
                last_avg_loss = avg_loss

            if debug:
                print(f"Época [{epoch+1}/{self.epochs}] - Pérdida: {avg_loss:.6f} - Perdida a comparar {last_avg_loss},"
                      f" Counter: {counter}")

        self.trained = True

    def transform(self, data: np.ndarray) -> torch.Tensor:
        """
        Metodo encargado de calcular y devolver los embeddings de los datos recibidos
        """
        if not self.trained:
            raise NotFittedError("El autoencoder no ha sido entrenado aún.")

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            embedding = self.encode(data)
        return embedding
