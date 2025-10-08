from packages import MixedManifoldDetector
from packages.autoencoders import LinearAutoencoder

ae = LinearAutoencoder.LinearAutoencoder(input_dim=23)
md = MixedManifoldDetector.MixedManifoldDetector(autoencoder=ae)
