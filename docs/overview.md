# Project Overview: NISCA

**NISCA (Nonlinear Identifiable Simplex Component Analysis)** is a deep generative framework for unsupervised latent source separation in high-dimensional imaging data. It is designed to disentangle structured latent sources under nonlinear mixing and observational noise, with applications in:

- Hyperspectral remote sensing
- Dynamic contrast-enhanced MRI
- Financial time series modeling

The method extends the variational autoencoder (VAE) framework by introducing:
- Simplex-constrained latent priors (Dirichlet, Logistic-Normal)
- Invertible nonlinear decoders for post-nonlinear mixtures
- Identifiability and interpretability under theoretical guarantees
