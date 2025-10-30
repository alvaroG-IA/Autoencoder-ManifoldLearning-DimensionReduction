import argparse
import pandas as pd
import numpy as np
import torch
from packages.Utils import seleccionar_autoencoder, seleccionar_manifold, generate_plot, set_global_seed
from packages.MixedManifoldDetector import MixedManifoldDetector
from packages.autoencoders.LinearAutoencoder import LinearAutoencoder
from packages.autoencoders.LinearSparseAutoencoder import LinearSparseAutoencoder
from packages.autoencoders.DenoisingSparseAutoencoder import DenoisingSparseAutoencoder
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler

DATA_SCALED = True
GLOBAL_SEED = 42
set_global_seed(GLOBAL_SEED)


def main():
    # -- ARGUMENTOS ESPERADOS --
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv_path", help="Ruta del .cvs que se utilizará para entrenar el sistema", type=str)
    parser.add_argument("test_csv_path", help="Ruta del .csv que se utilizara como datos de test del sistema", type=str)
    args = parser.parse_args()

    # -- CARGA DE CONJUNTOS DE DATOS --
    train_df = pd.read_csv(args.train_csv_path)
    test_df = pd.read_csv(args.test_csv_path)

    # -- SEPARACION DE LABELS --
    train_labels = np.array(train_df.iloc[:, 0]).astype(int)
    train_data = np.array(train_df.iloc[:, 1:])

    test_labels = np.array(test_df.iloc[:, 0]).astype(int)
    test_data = np.array(test_df.iloc[:, 1:])

    # -- COMPROBAMOS QUE LAS DIMENSIONES DE LOS DATOS DE TRAIN Y TEST SEAN IGUALES
    _, train_shape = train_data.shape
    _, test_shape = test_data.shape
    if train_shape != test_shape:
        raise ValueError('Train and test shapes do not match')

    if DATA_SCALED:
        scler = MinMaxScaler()
        train_data = scler.fit_transform(train_data)
        test_data = scler.transform(test_data)

    # -- PARÁMETROS COMPARTIDOS POR LOS AUTOENCODERS --
    input_dim = train_data.shape[1]
    embedding_dim = 32
    epochs = 200
    loss_threshold = 1e-4
    min_delta = 1e-3
    max_num_iters_without_progress = 30
    batch_size = 32
    optimizer_class = torch.optim.Adam
    lr = 1e-3
    loss_fn = torch.nn.MSELoss()
    data_scaled = DATA_SCALED
    lamda_l1 = 1e-4
    noise_factor = 0.3

    # -- SELECCIÓN TIPO DE AUTOENCODER
    opt = seleccionar_autoencoder()
    if opt == 2:
        autoencoder_type = "LinearSparseAutoencoder"
        autoencoder = LinearSparseAutoencoder(
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

    elif opt == 3:
        autoencoder_type = 'DenoisingSparseAutoencoder'
        autoencoder = DenoisingSparseAutoencoder(
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
        autoencoder_type = 'LinearAutoencoder'
        autoencoder = LinearAutoencoder(
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

    # -- SELECCIÓN ALGORITMO DE MANIFOLD LEARNING
    opt = seleccionar_manifold()
    if opt == 1:
        manifold_model = 't-SNE'
        manifold_alg = TSNE(n_components=2, perplexity=20, verbose=1,
                            n_iter_without_progress=50, random_state=GLOBAL_SEED)
    else:
        manifold_model = 'LLE'
        manifold_alg = LocallyLinearEmbedding(n_neighbors=5, n_components=2, random_state=GLOBAL_SEED)

    mx = MixedManifoldDetector(autoencoder=autoencoder, manifold_alg=manifold_alg, seed=GLOBAL_SEED)

    pts_2d_train = mx.fit_transform(data=train_data)

    pts_2d_test = mx.transform(data=test_data, k=5)

    # --- GENERAMOS LOS PLOTS PARA EL CONJUNTO DE ENTRENAMIENTO ---

    data_name_train = args.train_csv_path.split('.')[0]
    data_name_test = args.test_csv_path.split('.')[0]

    generate_plot(pts=pts_2d_train, labels=train_labels,
                  data_name=data_name_train, manifold_model=manifold_model,
                  autoencoder_model=autoencoder_type)

    generate_plot(pts=pts_2d_train,
                  data_name=data_name_train, manifold_model=manifold_model,
                  autoencoder_model=autoencoder_type)

    # --- GENERAMOS LOS PLOTS PARA EL CONJUNTO DE TEST ---

    generate_plot(pts=pts_2d_test, labels=test_labels,
                  data_name=data_name_test, manifold_model=manifold_model,
                  autoencoder_model=autoencoder_type)

    generate_plot(pts=pts_2d_test,
                  data_name=data_name_test, manifold_model=manifold_model,
                  autoencoder_model=autoencoder_type)


if __name__ == "__main__":
    main()
