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
    parser.add_argument("train_csv_path", help="Ruta del .cvs que se utilizará para entrenar el sistema", type=str)
    parser.add_argument("test_csv_path", help="Ruta del .csv que se utilizara como datos de test del sistema", type=str)
    args = parser.parse_args()
    """
    train_path = 'data/FASHION_MNIST/fashion-mnist_train.csv'
    test_path = 'data/FASHION_MNIST/fashion-mnist_test.csv'
    """

    train_df = pd.read_csv(args.train_csv_path)
    test_df = pd.read_csv(args.test_csv_path)

    train_labels = np.array(train_df['label'])
    train_data = np.array(train_df.drop('label', axis=1))

    test_labels = np.array(test_df['label'])
    test_data = np.array(test_df.drop('label', axis=1))

    train_shape = train_data.shape
    test_shape = test_data.shape
    if train_shape[1] != test_shape[1]:
        raise ValueError('Train and test shapes do not match')

    if DATA_SCALED:
        train_data = train_data / 255
        test_data = test_data / 255

    input_dim = train_data.shape[1]
    embedding_dim = 32
    epochs = 100
    loss_threshold = 1e-3
    batch_size = 64
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
    pts_2d_train = mx.fit_transform(data=train_data)
    pts_2d_test = mx.transform(data=test_data, k=10)

    # --- GENERAMOS LOS PLOTS PARA EL CONJUNTO DE ENTRENAMIENTO ---

    plt.scatter(pts_2d_train[:, 0], pts_2d_train[:, 1], c=train_labels, cmap='jet', s=1)
    plt.title(f'MIXED MANIFOLD DETECTION {MANIFOLD_MODEL}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Digit')
    plt.savefig(f"{args.train_csv_path}_{MANIFOLD_MODEL}_labels.png")
    plt.close()

    plt.scatter(pts_2d_train[:, 0], pts_2d_train[:, 1], s=1)
    plt.title(f'MIXED MANIFOLD DETECTION {MANIFOLD_MODEL}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(f"{args.train_csv_path}_{MANIFOLD_MODEL}.png")
    plt.close()

    # --- GENERAMOS LOS PLOTS PARA EL CONJUNTO DE TEST ---

    plt.scatter(pts_2d_test[:, 0], pts_2d_test[:, 1], c=test_labels, cmap='jet', s=1)
    plt.title(f'MIXED MANIFOLD DETECTION {MANIFOLD_MODEL}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Digit')
    plt.savefig(f"{args.test_csv_path}_{MANIFOLD_MODEL}_labels.png")
    plt.close()

    plt.scatter(pts_2d_test[:, 0], pts_2d_test[:, 1], s=1)
    plt.title(f'MIXED MANIFOLD DETECTION {MANIFOLD_MODEL}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(f"{args.test_csv_path}_{MANIFOLD_MODEL}.png")
    plt.close()


if __name__ == "__main__":
    main()
