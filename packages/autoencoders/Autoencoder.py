import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod


class Autoencoder(nn.Module, ABC):
    """
    Clase abstracta que sirve de interfaz para multiples implementaciones de distintos tipos de autoencoders
    """
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 32,
                 epochs: int = 100,
                 loss_threshold: float = 0,
                 batch_size: int = 32,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 lr: float = 1e-3,
                 loss_fn: torch.nn.MSELoss = nn.MSELoss(),
                 data_scaled: bool = False):

        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.loss_threshold = loss_threshold
        self.batch_size = batch_size
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.loss_fn = loss_fn
        self.data_scaled = data_scaled
        self.trained = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        pass


    @abstractmethod
    def fit(self, data: np.ndarray, min_delta: float = 1e-6, max_num_iters_without_progress: int = 20, debug: bool = False):
        """
        Método encargado de llevar a cabo el entrenamiento del autoencoder
        :param data: datos utilizados para el entrenamiento
        :param debug: parametro para decidir si se muestra por pantalla los resultados del entrenamiento
        :param min_delta: valor minimo para el entrenamiento #####
        :param max_num_iters_without_progress: valor minimo para el entrenamiento #####
        :return:
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> torch.Tensor:
        """
        Metodo encargado de hacer uso del autoencoder y generar los embeddings
        :param data: datos a transformar
        :return:
        """
        pass
