import argparse
import pandas as pd
import numpy as np
from packages import MixedManifoldDetector
from packages.autoencoders import LinearAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Ruta del .cvs a procesar", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)


if __name__ == "__main__":
    main()
