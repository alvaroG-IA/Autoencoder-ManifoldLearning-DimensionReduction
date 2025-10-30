import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def seleccionar_autoencoder():
    """
    Función encargada de permitir al usuario seleccionar el tipo de autoencoder a utilizar mediante un menu por terminal
    """
    while True:
        print("[Menú selección de modelo Autoencoder]")
        print("[1] Autoencoder Lineal")
        print("[2] Autoencoder Lineal con regularización Sparse (L1)")
        print("[3] Denoising Autoencoder con regularización Sparse (ruido + L1)")
        opt = input("Opción a elegir: ").strip()

        if opt in {"1", "2", "3"}:  # Si la opción elegida es válida:
            modelos = {
                "1": "Autoencoder Lineal",
                "2": "Autoencoder Lineal con regularización Sparse (L1)",
                "3": "Denoising Autoencoder con regularización Sparse (ruido + L1)"
            }
            print(f"✅ Seleccionado: {modelos[opt]}")
            return int(opt)
        else:   # si la opción seleccionada no es válida:
            print("⚠️  Opción no válida, intenta de nuevo...")
            time.sleep(0.5)
            os.system("clear" if os.name != "nt" else "cls")


def seleccionar_manifold():
    """
    Función encargada de permitir al usuario seleccionar el algoritmo de manifold learning
    a utilizar mediante un menu por terminal
    """
    while True:
        print("[Menú selección de método Manifold Learning]")
        print("[1] t-SNE")
        print("[2] Locally Linear Embedding (LLE)")
        opt = input("Opción a elegir: ").strip()

        if opt in {"1", "2"}:
            modelos = {
                "1": "t-SNE",
                "2": "Locally Linear Embedding (LLE)"
            }
            print(f"✅ Seleccionado: {modelos[opt]}")
            return int(opt)
        else:
            print("⚠️  Opción no válida, intenta de nuevo...")
            time.sleep(0.5)
            os.system("clear" if os.name != "nt" else "cls")


def generate_plot(pts, data_name: str, manifold_model: str, autoencoder_model: str, labels=None):
    """
    Funcion encargada de generar la visualización de los datos que recibe en un archivo png
    :param pts: puntos que van a ser mostrados en el gráfico
    :param data_name: nombre del archivo de los datos
    :param manifold_model: nombre del modelo de manifold usado
    :param autoencoder_model:  nombre del tipo de autoencoder usado
    :param labels: conjunto de etiquetas de los datos
    :return:
    """
    plt.figure(figsize=(6, 6))

    if labels is not None:  # caso en el que la funciór reciba un conjunto de etiquetas:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        cmap = plt.get_cmap('tab10', n_classes)  # selección de colores deterministas en base al numero de labels
        norm = BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)

        scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap=cmap, norm=norm, s=5)
        plt.title(f'MIXED MANIFOLD DETECTION {manifold_model} (ANOTADO)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        cbar = plt.colorbar(scatter, ticks=range(n_classes))
        cbar.set_label('Label')
        cbar.set_ticklabels(unique_labels)
        plt.savefig(f"{data_name}_{autoencoder_model}_{manifold_model}_labels.png", dpi=300)
        plt.close()
    else:
        plt.scatter(pts[:, 0], pts[:, 1], s=5)
        plt.title(f'MIXED MANIFOLD DETECTION {manifold_model}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(f"{data_name}_{autoencoder_model}_{manifold_model}.png", dpi=300)
        plt.close()


def set_global_seed(seed: int = 42):
    """
    Funcion encargada de establecer una única semilla global para la ejecución
    :param seed: valor de semilla seleccionado. Por defecto será igual a 42.
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'--- Semilla establecida en el valor de {seed} ---')
