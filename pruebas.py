import pandas as pd
import numpy as np
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder
from packages.MixedManifoldDetector import MixedManifoldDetector
import torch
import matplotlib.pyplot as plt


data_path = 'data/MNIST/mnist_test.csv'
raw = pd.read_csv(data_path)

labels = np.array(raw['label'])
data = np.array(raw.drop(['label'], axis=1)) / 256

input_dim = data.shape[1]
embedding_dim = 32
epochs = 100
loss_threshold = 1e-3
batch_size = 32
optimizer_class = torch.optim.Adam
lr = 1e-3
loss_fn = torch.nn.MSELoss()

ae = LinearAutoencoder(
    input_dim=input_dim,
    embedding_dim=embedding_dim,
    epochs=epochs,
    loss_threshold=loss_threshold,
    batch_size=batch_size,
    optimizer_class=optimizer_class,
    lr=lr,
    loss_fn=loss_fn)

mx = MixedManifoldDetector(autoencoder=ae)
pts_2d = mx.fit_transform(data=data)

plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=labels, cmap='jet', s=1)
plt.title('MIXED MAINFOLD DETECTION')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Digit')
plt.savefig("pruebaMNIST.png")
