import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from packages.Utils import generate_plot
from packages.MixedManifoldDetector import MixedManifoldDetector
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder
from packages.autoencoders.LinearSparseAutoencoder import LinearSparseAutoencoder
from packages.autoencoders.DenoisingSparseAutoencoder import DenoisingSparseAutoencoder
from sklearn.manifold import TSNE, LocallyLinearEmbedding

DATA_SCALED = True
MANIFOLD_MODEL = 'TSNE'
AUTOENCODER_TYPE = 'LinearAutoencoder'


def main():
    global AUTOENCODER_TYPE

    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv_path", help="Ruta del .cvs que se utilizará para entrenar el sistema", type=str)
    parser.add_argument("test_csv_path", help="Ruta del .csv que se utilizara como datos de test del sistema", type=str)
    args = parser.parse_args()

    """
    train_path = 'data/CIFAR-10/train.csv'
    test_path = 'data/CIFAR-10/test.csv'
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
    loss_threshold = 1e-4
    min_delta = 1e-3
    max_num_iters_without_progress = 30
    batch_size = 64
    optimizer_class = torch.optim.Adam
    lr = 1e-3
    loss_fn = torch.nn.MSELoss()
    data_scaled = DATA_SCALED
    lamda_l1 = 1e-4
    noise_factor = 0.25

    if AUTOENCODER_TYPE == 'LinearSparseAutoencoder':
        print('Using LinearSparseAutoencoder')
        ae = LinearSparseAutoencoder(
            lambda_l1=lamda_l1,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            epochs=epochs,
            loss_threshold=loss_threshold,
            min_delta=min_delta,
            max_num_iters_without_progress=max_num_iters_without_progress,
            batch_size=batch_size,
            optimizer_class=optimizer_class,
            lr=lr,
            loss_fn=loss_fn,
            data_scaled=data_scaled
        )

    elif AUTOENCODER_TYPE == 'DenoisingSparseAutoencoder':
        print('Using DenoisingSparseAutoencoder')
        ae = DenoisingSparseAutoencoder(
            noise_factor=noise_factor,
            lambda_l1=lamda_l1,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            epochs=epochs,
            loss_threshold=loss_threshold,
            min_delta=min_delta,
            max_num_iters_without_progress=max_num_iters_without_progress,
            batch_size=batch_size,
            optimizer_class=optimizer_class,
            lr=lr,
            loss_fn=loss_fn,
            data_scaled=data_scaled
        )

    else:
        print('Using LinearAutoencoder')
        AUTOENCODER_TYPE = 'LinearAutoencoder'
        ae = LinearAutoencoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            epochs=epochs,
            loss_threshold=loss_threshold,
            min_delta=min_delta,
            max_num_iters_without_progress=max_num_iters_without_progress,
            batch_size=batch_size,
            optimizer_class=optimizer_class,
            lr=lr,
            loss_fn=loss_fn,
            data_scaled=data_scaled
        )

    if MANIFOLD_MODEL == 'TSNE':
        manifold_alg = TSNE(n_components=2, perplexity=15, n_iter_without_progress=50)
    else:
        manifold_alg = LocallyLinearEmbedding(n_neighbors=10, n_components=2)

    mx = MixedManifoldDetector(autoencoder=ae, manifold_alg=manifold_alg)
    pts_2d_train = mx.fit_transform(data=train_data)
    pts_2d_test = mx.transform(data=test_data, k=5)

    # --- GENERAMOS LOS PLOTS PARA EL CONJUNTO DE ENTRENAMIENTO ---

    generate_plot(pts=pts_2d_train, labels=train_labels,
                  data_name=args.train_csv_path, manifold_model=MANIFOLD_MODEL,
                  autoencoder_model=AUTOENCODER_TYPE)

    generate_plot(pts=pts_2d_train,
                  data_name=args.train_csv_path, manifold_model=MANIFOLD_MODEL,
                  autoencoder_model=AUTOENCODER_TYPE)

    # --- GENERAMOS LOS PLOTS PARA EL CONJUNTO DE TEST ---

    generate_plot(pts=pts_2d_test, labels=test_labels,
                  data_name=args.test_csv_path, manifold_model=MANIFOLD_MODEL,
                  autoencoder_model=AUTOENCODER_TYPE)

    generate_plot(pts=pts_2d_test,
                  data_name=args.test_csv_path, manifold_model=MANIFOLD_MODEL,
                  autoencoder_model=AUTOENCODER_TYPE)

if __name__ == "__main__":
    main()
