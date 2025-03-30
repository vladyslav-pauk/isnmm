# NISCA: Nonlinear Identifiable SCA

This repository provides an end-to-end implementation of the Nonlinear Identifiable Simplex Constrained Autoencoder (NISCA).
NISCA is a deep learning algorithm for estimating categorical latent factor from noisy multivariate data, with applications in hyperspectral remote sensing, medical imaging, and finance.

[//]: # (It includes comprehensive [documentation]&#40;docs/index.md&#41;, reproducible experiment [configurations]&#40;docs/configuration.md&#41;, and interactive [notebooks]&#40;#usage&#41; for demonstration and analysis.  )

[//]: # (This codebase delivers a modular, scalable, and production-ready implementation of the NISCA model.  )


[//]: # (The model leverages variational inference with simplex-constrained latent priors to enable interpretable and theoretically identifiable representations under nonlinear mixing and observational noise. )

[//]: # (This package provides an end-to-end implementation of the NISCA framework.)

[//]: # (- Provides unsupervised latent source separation under nonlinear mixing and noise.)

[//]: # (- Trains deep generative models with interpretable latent representations using simplex constraints.)

[//]: # (- Enables theoretical identifiability for nonlinear ICA settings.)

[//]: # (- Scalable training pipeline with CUDA acceleration and full experiment tracking.)

[//]: # (Welcome to )
[//]: # (This repository provides a production-grade implementation of NISCA — a deep generative framework for modeling multivariate data with categorical priors.)
[//]: # (Designed for unsupervised source separation, if finds most prominent applications in hyperspectral remote sensing, medical imaging &#40;e.g., DCE-MRI&#41;, and financial data modeling.)
[//]: # (a probabilistic model for spectral and multi-channel images.)

[//]: # (# Project Overview: NISCA)

[//]: # (**NISCA &#40;Nonlinear Identifiable Simplex Component Analysis&#41;** is a deep generative framework for unsupervised latent source separation in high-dimensional imaging data. It is designed to disentangle structured latent sources under nonlinear mixing and observational noise, with applications in:)

[//]: # ()
[//]: # (- Hyperspectral remote sensing)

[//]: # (- Dynamic contrast-enhanced MRI)

[//]: # (- Financial time series modeling)

[//]: # ()
[//]: # (The method extends the variational autoencoder &#40;VAE&#41; framework by introducing:)

[//]: # (- Simplex-constrained latent priors &#40;Dirichlet, Logistic-Normal&#41;)

[//]: # (- Invertible nonlinear decoders for post-nonlinear mixtures)

[//]: # (- Identifiability and interpretability under theoretical guarantees)

[//]: # (, aimed at latent source identification.)

[//]: # (It is designed for both research and applied settings, particularly in domains such as hyperspectral remote sensing and dynamic medical imaging &#40;e.g., DCE-MRI&#41;.)

[//]: # (Designed for scalability and modularity, the codebase supports high-performance CUDA-accelerated training and inference and structured experiment orchestration.)

[//]: # (flexible experimentation with model architectures, training configurations, and datasets.)

[//]: # (The framework is designed for scalability and modularity, allowing for easy experimentation with different model architectures and configurations or datasets.)

[//]: # (This repository contains a production-grade implementation, along with exhaustive documentation, and demonstration notebooks.)

[//]: # (Welcome to **NISCA** — a deep generative modeling framework for multivariate data with categorical latent structure and source separation under nonlinear mixing.  )

[//]: # (The framework is designed for both research and applied settings, particularly in domains such as **hyperspectral remote sensing** and **dynamic medical imaging** &#40;e.g., DCE-MRI&#41;.)

[//]: # (Built with scalability and modularity in mind, the codebase supports flexible experimentation with alternative model architectures, training configurations, and datasets.)



[//]: # (The framework is designed to support flexible experimentation across a range of model architectures, training regimes, and datasets.)
[//]: # (It supports both hyperspectral and medical data formats, includes synthetic simulation pipelines, and uses structured experiment tracking.)


[//]: # (## Highlights)
[//]: # (- **Variational autoencoders** with simplex-constrained latent priors &#40;Dirichlet, Logistic-Normal&#41;)
[//]: # (- **Post-nonlinear decoder** supporting arbitrary invertible transforms)
[//]: # (- **Synthetic + real-world** data support &#40;Urban, Cuprite, MRI, financial datasets&#41;)
[//]: # (- **Multi-experiment orchestration**, WandB logging, sweeping, and GCP/Docker compatibility)
[//]: # (- **Metrics for identifiability and recovery**: subspace distance, Amari index, mutual info, etc.)

[//]: # (This repository provides the full codebase, configurations, data utilities, and evaluation framework for **NISCA**, a probabilistic model for **nonlinear, unsupervised source separation** in high-dimensional imaging data, such as hyperspectral images and DCE-MRI scans.)
[//]: # ()
[//]: # (The model is based on a constrained variational autoencoder framework with geometric priors &#40;e.g., Dirichlet&#41; and nonlinear decoders, achieving **identifiable and interpretable latent representations** even in complex post-nonlinear mixtures.)


[//]: # (geometric priors, e.g. Dirichlet, and nonlinear decoders)
[//]: # (complex post-nonlinear mixtures)
[//]: # (tissue and material separation from high-dimensional imaging data)

## Contents

- [Overview](#overview)

- [Getting Started](#getting started)

- [Usage](#usage)

- [Documentation](#documentation)

- [Citing](#citing)

- [Contributing](#contributing)

- [License](#license)

- [Contact](#contact)

[//]: # (- [Contact]&#40;#contact&#41;)

## Overview

### Architecture Design
NISCA extends traditional variational autoencoder architecture with domain-specific inductive bias, enabling identifiable latent representations under noisy nonlinear mixing with theoretical guarantees.
By using geometrically constrained simplex priors, NISCA facilitates unsupervised latent source separation under categorical priors, also known as simplex component analysis (SCA).

Key properties include:
- **Bayesian** inference via deep variational autoencoders (VAEs)
- Geometrically constrained latent space (simplex priors) suitable for **categorical** ground truth
- Trainable **post-nonlinear decoder** supporting arbitrary invertible transforms
- Theoretical **identifiability** under nonlinear mixing and noise

### Implementation:

This production-grade implementation is built on top of PyTorch Lightning engine and facilitates training orchestration and systematic model evaluation.

Key features include:
- **Synthetic and real-world** data support (Urban, Cuprite, MRI, financial datasets)
- Comprehensive **experiment tracking** with Weights & Biases and Tensorboard logs.
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

### Performance Summary

NISCA is applicable to a range of real-world tasks, including spectral unmixing in hyperspectral remote sensing, tissue segmentation in dynamic contrast-enhanced MRI, and factor modeling in financial portfolios.

Key results include:
- Achieves **~20% improvement** in latent factor estimation over CNAE/NMF
- Trains **2× faster** than CNAE benchmark
- Improved **interpretability** and class **separability**
- Provides **theoretical identifiability guarantees** for nonlinear mixing

[//]: # (The model achieves:)
[//]: # (- 2× faster training convergence with constrained latent space)
[//]: # (- Recovers **interpretable** latent factors)
[//]: # (- Strong generalization to unseen imaging samples)

## Getting Started

### Interactive Notebooks
The repository includes Jupyter notebooks for interactive exploration of the models and datasets and reproducing results from the paper.

[//]: # (Explore interactive Jupyter notebooks covering a range of topics from training to model evaluation and visualization, as well as some experiments.&#41;)

- [**Model Training**](notebooks/model_training.ipynb): Walkthrough of training a model with different configurations.
- [**Training Evaluation**](notebooks/quantitative_evaluation.ipynb): Benchmarking against classic methods and analyzing results. 
- [**Synthetic Data**](notebooks/mixture_model.ipynb): Generating and visualizing synthetic data for testing.
- Posterior sampling and latent space visualization
- Component-wise reconstruction

[//]: # (- Quantitative evaluation and benchmark comparisons)
[//]: # (- Model training walkthrough)
[//]: # (- Posterior sampling and latent space visualization)
[//]: # (Sweep summary and performance evaluation)
[//]: # (Available in notebooks/ and notebooks/figures/.)

### Installation and Deployment

To run repository locally or in a cloud environment, follow the instructions in the [installation guide](docs/installation.md).

## Usage





### Run Experiments

To run a hyperparameter sweep or schedule multiple experiments, execute

```bash
  PYTHONPATH=./ python src/scripts/run_sweep.py --experiment synthetic --sweep test_run
```

All outputs, including checkpoints, logs, and evaluation results, will be stored under `experiments/{experiment_name}/config/sweep/`.

Find more information on experiments [here](nisca.wiki/experiments).

### Explore Results

To access results of the latest training run, use:

```bash
  PYTHONPATH=./ python src/scripts/explore_model.py --experiment synthetic
```
or pass with `--run_id <id>` flag to visualize a specific run.
Plots and logs will be saved and logged to W&B under the specified experiment name.


To analyze the latest sweep, run:

```bash
  PYTHONPATH=./ python src/scripts/analyze_sweep.py --experiment synthetic
```
or pass with `--sweep <sweep_name>` flag to visualize a specific sweep.

### CUDA Support

To enable CUDA support, set the environment variable `CUDA_VISIBLE_DEVICES` to the desired GPU ID(s):

```bash
  export CUDA_VISIBLE_DEVICES=0,1
```

You can run them in a Jupyter environment or convert them to scripts using `nbconvert`.

```bash
  jupyter nbconvert --to script notebook.ipynb
```

## Documentation

[//]: # (For more details refer to [documentation]&#40;docs/index.md&#41;.)

Learn more about the NISCA framework, its components, and how to use it effectively by exploring the documentation:

- [Model Overview](docs/model.md)
- [Evaluation Metrics](docs/metrics.md)
- [Datasets](docs/datasets.md)
- [Experiments](docs/experiments.md)
- [Configuration](docs/configuration.md)
- [Implementation](docs/implementation.md)

An in-depth account of the theoretical framework and empirical evaluation is provided in the following publications:

- [Master's Thesis](pubs/thesis.pdf): *Deep Generative Modeling for Hyperspectral & Medical Imaging*, Vladyslav Pauk, OSU (2024).
- **IEEE Preprint**: Coming soon, see draft [here](pubs/preprint.pdf).



## Citing

This work is being prepared for submission/under review in IEEE Signal Processing.
Please cite as: 
```bibtex
@misc{pauk2024generative,
  author = {Vladyslav Pauk},
  title = {Post-Nonlinear Mixture Identification via Variational Auto-Encoding},
  year = {2024},
  note = {Master's Thesis, Oregon State University}
}
```

If you use this code, please cite this repository:
```bibtex
@misc{pauk2024nisca,
    title = {NISCA: Nonlinear ICA with Simplex Priors},
    author = {Vladyslav Pauk},
    howpublished = {\url{https://github.com/vladyslav-pauk/nisca}},
    year = {2025},
    note = {GitHub repository}
}
```


## Acknowledgements

This work was initiated as part of the Master's thesis research of Dr. Vladyslav Pauk under the supervision of Prof. Xiao Fu at Oregon State University.


[//]: # (## Contact)

[//]: # ()
[//]: # (- **Author**: Dr. Vladyslav Pauk  )

[//]: # (- **Email**: [paukvp@gmail.com]&#40;mailto:paukvp@gmail.com&#41;  )

[//]: # (- **Website**: [linkedin.com/vladyslav-pauk]&#40;https://www.linkedin.com/in/vladyslav-pauk&#41;)


## Contributing

Pull requests, feedback, and discussions are welcome. Please submit issues or suggestions via GitHub.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For academic or technical questions, please contact Dr. [Vladyslav Pauk](mailto:paukvp@gmail.com).