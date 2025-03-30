## Model Overview

A scalable, interpretable latent variable model for unsupervised tissue and material separation from high-dimensional imaging data, including hyperspectral satellite images and DCE-MRI scans. This work extends post-nonlinear ICA using variational autoencoders (VAEs) with geometric constraints.

### Motivation

Traditional methods like PCA, ICA, or NMF fail under nonlinear mixing, noise, and non-identifiable settings common in real-world medical and satellite data. This project addresses these limitations by combining:
- Variational inference
- Geometric priors (e.g., simplex)
- Deep nonlinear decoder
- Theoretical guarantees on identifiability



### Problem

Estimate latent sources \(\mathbf{Z} \in \mathbb{R}^{N \times K}\) from observed data \(\mathbf{X} \in \mathbb{R}^{N \times D}\) under:
- Nonlinear mixture: \(\mathbf{X} = f(\mathbf{Z}) + \varepsilon\)
- Simplex-constrained latent space: \(\mathbf{Z} \in \Delta^K\)
- No supervision or ground truth



### Model Design

- **Encoder**: Learns variational posterior \(q_\phi(\mathbf{z} \mid \mathbf{x})\)
- **Latent Prior**: Dirichlet or Logistic-Normal (simplex geometry)
- **Decoder**: MLP with nonlinearity \(f_\theta(\mathbf{z})\)
- **Objective**: ELBO with identifiability/sparsity regularization

\[
\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - \text{KL}[q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z})]
\]



### Training Pipeline

- **Framework**: PyTorch Lightning
- **Experiment Tracking**: Weights & Biases
- **Reproducibility**: Docker + GCP
- **Techniques**: Early stopping, MC sampling, learning rate warmup



### Results

- ~20% improvement in latent recovery vs baselines (PCA, ICA, NMF, VASCA)
- 2× faster convergence
- First **identifiability result** for nonlinear simplex component analysis
- Applications: tumor separation in DCE-MRI, mineral mapping in hyperspectral imagery


### Data

- Synthetic hyperspectral benchmarks (ground truth known)
- 192-band satellite imagery (Urban, Cuprite)
- DCE-MRI scans (anonymized, simulated)



### Key Concepts

- Post-Nonlinear ICA
- Simplex Component Analysis (SCA)
- Variational Inference & Latent Priors
- Geometric Identifiability



### Future Work

- Integrate spatial priors (e.g., CRFs)
- Uncertainty quantification in diagnosis
- Deployment with TorchScript for real-time inference



[//]: # ()
[//]: # (# Non-linear Auto-Encoder &#40;NAE&#41;)

[//]: # ()
[//]: # (## CNAE)

[//]: # ()
[//]: # (Deterministic &#40;noiseless&#41; model Constrained Non-Linear Autoencoder:)

[//]: # ()
[//]: # ($$)

[//]: # (    \min_{\theta} \frac{1}{N} \sum_{i=1}^{N} \left\| f&#40;q&#40;x_i&#41;&#41; - x_i \right\|_2^2,\\)

[//]: # (    \text{ subject to } \bm 1^\top \bm q&#40;\bm x_i&#41; - 1 = 0 )

[//]: # ($$)

[//]: # ()
[//]: # (Optimized using augmented Lagrangian multiplier method.)

[//]: # ()
[//]: # (## SNAE &#40;NAES&#41;)

[//]: # ()
[//]: # (Deterministic &#40;noiseless&#41; model Nonlinear Autoencoder on a Simplex &#40;Reparametrized&#41;)

[//]: # ()
[//]: # (Same loss as CNAE, but instead of the constraint we use mapping onto simplex &#40;reparameterization&#41;.)

[//]: # ()
[//]: # (# Logistic Variational Auto-Encoder &#40;VASCA&#41;)

[//]: # ()
[//]: # (## VASCA)

[//]: # ()
[//]: # (## NISCA)

[//]: # ()
[//]: # (Add nonlinear layers to the decoder in VASCA.)

[//]: # ()
[//]: # (## INISCA)

[//]: # ()
[//]: # (Invertible NISCA with PNL encoder.)

[//]: # ()
[//]: # (# Log-ratio Variational Auto-Encoder &#40;NIVA&#41;)

[//]: # ()
[//]: # (## NIVA)

[//]: # ()
[//]: # (Aitchison distribution Encoder)

[//]: # ()
[//]: # (## CNIVA)

[//]: # ()
[//]: # (Constrained Non-Linear Identifiable Variational Autoencoder)

[//]: # ()
[//]: # (## INIVA)

[//]: # ()
[//]: # (Variational Auto-Encoder on a Convex Hull &#40;CHVAE&#41;)