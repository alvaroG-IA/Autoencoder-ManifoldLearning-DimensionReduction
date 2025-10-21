import matplotlib.pyplot as plt


def generate_plot(pts, data_name: str, manifold_model: str, autoencoder_model: str, labels=None):
    if labels is not None:
        plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='jet', s=1)
        plt.title(f'MIXED MANIFOLD DETECTION {manifold_model} (ANOTADO)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(label='Digit')
        plt.savefig(f"{data_name}_{autoencoder_model}_{manifold_model}_labels.png")
        plt.close()
    else:
        plt.scatter(pts[:, 0], pts[:, 1], s=1)
        plt.title(f'MIXED MANIFOLD DETECTION {manifold_model}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(f"{data_name}_{autoencoder_model}_{manifold_model}.png")
        plt.close()

