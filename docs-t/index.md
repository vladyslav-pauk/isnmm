# NISCA: Documentation

NISCA is a deep generative modeling framework for unsupervised latent source separation in spectral imaging data, with applications in remote sensing, medical imaging, and finance.
It extends traditional variational autoencoder architecture with the domain-specific inductive bias, by using geometrically constrained simplex priors, enabling interpretable and identifiable latent representations under noisy nonlinear mixing.

### Model Overview:
- **Bayesian** inference via deep variational autoencoders (VAEs)
- Geometrically constrained latent space (simplex priors) suitable for **categorical** ground truth
- Trainable **post-nonlinear decoder** supporting arbitrary invertible transforms
- Theoretical **identifiability** under nonlinear mixing and noise

### Features:
- **Synthetic and real-world** data support (Urban, Cuprite, MRI, financial datasets)
- Comprehensive **experiment tracking** with Weights & Biases
- **Efficient and scalable**: PyTorch Lightning pipeline and high-performance computing using CUDA
- **Multi-experiment orchestration**, streamlined sweeping, logging, and model serialization

[//]: # (- **Modular and scalable** PyTorch Lightning pipeline, integrated with Docker for cloud deployment)
[//]: # (- **Metrics for identifiability and parameter recovery**: subspace distance, Amari index, mutual info, etc.)

[//]: # (### Technological Stack)
[//]: # (- **PyTorch Lightning** for training and evaluation)
[//]: # (- **Weights & Biases &#40;W&B&#41;** for logging)
[//]: # (- **NumPy**, **Matplotlib**, **Scikit-learn**)
[//]: # (- **Docker** + **GCP** compatibility)
[//]: # (- **Configurable JSON experiments**)
[//]: # (- Optional **CUDA** acceleration)

### Empirical Results

- Achieves **~20% improvement** in latent factor estimation over nICA/NMF
- Trains **2× faster** than constrained benchmarks
- Improved **interpretability** and class **separability**
- Provides **theoretical identifiability guarantees** for nonlinear mixing

[//]: # (The model achieves:)
[//]: # (- 2× faster training convergence with constrained latent space)
[//]: # (- Recovers **interpretable** latent factors)
[//]: # (- Strong generalization to unseen imaging samples)

## Contents

- [Installation](installation.md)
- [Model Overview](formalism.md)
- [Implementation Details](implementation.md)
- [Experiments](experiments.md)
- [Configuration](configuration.md)