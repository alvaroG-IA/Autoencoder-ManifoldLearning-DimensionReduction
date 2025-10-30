from packages.autoencoders.Autoencoder import Autoencoder
import torch.nn as nn


class LinearAutoencoder(Autoencoder):
    """
    ---
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),     # Capa Linear Nº1
            nn.ReLU(),
            nn.Linear(256, 128),       # Capa Linear Nº2
            nn.ReLU(),
            nn.Linear(128, 64),        # Capa Linear Nº3
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),   # Capa generadora de embedding
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),

            nn.Sigmoid() if self.data_scaled else nn.Identity()        # añadimos una capa sigmoide para la salida
                                                                       # en el caso de que los datos hayan sido
                                                                       # estandarizados p
        )

    def forward(self, x):
        """
        Implementación del metodo abstracto `forward()` implementado en la clase padre
        :param x: datos de ejecución
        :return (embedding, recon): representación latente y reconstrucción de los datos de entrada
        """
        embedding = self.encoder(x)
        recon = self.decoder(embedding)
        return embedding, recon

    def encode(self, x):
        """
        Implementación del metodo abstracto `encode()` implementado en la clase padre
        :param x: datos de ejecución
        :return embedding: representación latente de los datos de entrada
        """
        return self.encoder(x)
