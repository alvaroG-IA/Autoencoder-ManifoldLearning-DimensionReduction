import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from packages.MixedManifoldDetector import MixedManifoldDetector
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder
from sklearn.manifold import TSNE, LocallyLinearEmbedding

DATA_SCALED = True
MANIFOLD_MODEL = 'TSNE'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Ruta del .cvs a procesar", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    labels = np.array(df['label'])
    data = np.array(df.drop('label', axis=1))

    if DATA_SCALED:
        data = data / 255

    input_dim = data.shape[1]
    embedding_dim = 32
    epochs = 200
    loss_threshold = 1e-3
    batch_size = 32
    optimizer_class = torch.optim.Adam
    lr = 1e-3
    loss_fn = torch.nn.MSELoss()
    data_scaled = DATA_SCALED

    ae = LinearAutoencoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        epochs=epochs,
        loss_threshold=loss_threshold,
        batch_size=batch_size,
        optimizer_class=optimizer_class,
        lr=lr,
        loss_fn=loss_fn,
        data_scaled=data_scaled
    )
    if MANIFOLD_MODEL == 'TSNE':
        manifold_alg = TSNE(n_components=2, perplexity=35, n_iter_without_progress=50)
    else:
        manifold_alg = LocallyLinearEmbedding(n_neighbors=10, n_components=2)

    mx = MixedManifoldDetector(autoencoder=ae, manifold_alg=manifold_alg)
    pts_2d = mx.fit_transform(data=data)

    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=labels, cmap='jet', s=1)
    plt.title(f'MIXED MANIFOLD DETECTION {MANIFOLD_MODEL}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Digit')
    plt.savefig(f"{args.csv_path}_{MANIFOLD_MODEL}_labels.png")
    plt.close()

    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], s=1)
    plt.title(f'MIXED MANIFOLD DETECTION {MANIFOLD_MODEL}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(f"{args.csv_path}_{MANIFOLD_MODEL}.png")
    plt.close()


if __name__ == "__main__":
    main()
