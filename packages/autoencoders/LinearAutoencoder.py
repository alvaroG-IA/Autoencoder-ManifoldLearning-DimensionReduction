import torch
from torch.utils.data import DataLoader
from packages.autoencoders.Autoencoder import Autoencoder
import numpy as np
import torch.nn as nn


class LinearAutoencoder(Autoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),     # Capa Linear Nº1
            nn.ReLU(),
            nn.Linear(128, 64),       # Capa Linear Nº2
            nn.ReLU(),
            nn.Linear(64, 64),        # Capa Linear Nº3
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),   # Capa generadora de embedding
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),

            nn.Sigmoid() if self.data_scaled else nn.Identity()        # añadimos una capa sigmoide para la salida
                                                                            # en el caso de que los datos hayan sido
                                                                            # estandarizados p
        )

    def forward(self, x: torch.Tensor):
        embedding = self.encoder(x)
        z = self.decoder(embedding)
        return embedding, z

    def fit(self, data: np.ndarray, debug: bool = False):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        self.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                embedding, z = self(batch)
                loss = self.loss_fn(z, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)

            if debug:
                print(f'Época: [{epoch+1}/{self.epochs}], Pérdida: {avg_loss:.4f}')

            if avg_loss < self.loss_threshold:
                print(f"El entrenamiento ha sido detenido en la época {epoch+1} dado que PERDIDA {avg_loss:.4f} < "
                      f"THRESHOLD {self.loss_threshold:.4f}")

    def transform(self, data: np.ndarray):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            embedding = self.encoder(data)
        return embedding
