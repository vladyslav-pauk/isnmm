# NISCA: Nonlinear Identifiable SCA

[//]: # (Welcome to )
This repository provides a production-grade implementation of NISCA — a deep generative modeling framework for multivariate data with categorical priors.

Nonlinearly Identifiable Simplex Constrained Autoencoder (NISCA) is a probabilistic model for nonlinear, unsupervised source separation in high-dimensional imaging data, such as spectral and multi-channel images 

[//]: # (, aimed at latent source identification.)

[//]: # (It is designed for both research and applied settings, particularly in domains such as hyperspectral remote sensing and dynamic medical imaging &#40;e.g., DCE-MRI&#41;.)

[//]: # (Designed for scalability and modularity, the codebase supports high-performance CUDA-accelerated training and inference and structured experiment orchestration.)

[//]: # (flexible experimentation with model architectures, training configurations, and datasets.)

[//]: # (The framework is designed for scalability and modularity, allowing for easy experimentation with different model architectures and configurations or datasets.)

[//]: # (This repository contains a production-grade implementation, along with exhaustive documentation, and demonstration notebooks.)

[//]: # (Welcome to **NISCA** — a deep generative modeling framework for multivariate data with categorical latent structure and source separation under nonlinear mixing.  )

[//]: # (The framework is designed for both research and applied settings, particularly in domains such as **hyperspectral remote sensing** and **dynamic medical imaging** &#40;e.g., DCE-MRI&#41;.)

[//]: # (Built with scalability and modularity in mind, the codebase supports flexible experimentation with alternative model architectures, training configurations, and datasets.)

[//]: # (This codebase delivers a modular, scalable, and production-ready implementation of the NISCA model.  )
[//]: # (It includes comprehensive documentation, reproducible experiment configurations, and interactive notebooks for demonstration and analysis.  )

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

- [Getting Started](#getting started)

- [Usage](#usage)

- [Documentation](#documentation)

- [Citing](#citing)

- [Contributing](#contributing)

- [License](#license)

[//]: # (- [Contact]&#40;#contact&#41;)

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

To train a model with a specific config file, e.g. `experiments/synthetic/config/test_run.yaml`, run:

```bash
  PYTHONPATH=./ python src/scripts/...
```

To run a hyperparameter sweep or schedule multiple experiments, execute

```bash
  PYTHONPATH=./ python src/scripts/run_sweep.py --experiment synthetic --sweep test_run
```

All outputs, including checkpoints, logs, and evaluation results, will be stored under `experiments/{experiment_name}/config/sweep/`.

Find more information on experiments [here](nisca.wiki/experiments).

### Explore Results

To analyze the latest sweep, run:

```bash
  PYTHONPATH=./ python src/scripts/analyze_sweep.py --experiment synthetic
```
or pass with `--sweep <sweep_name>` flag to visualize a specific sweep.

To access the latest run results, use:

```bash
  PYTHONPATH=./ python src/scripts/explore_model.py --experiment synthetic
```
or pass with `--run_id <id>` flag to visualize a specific run.

Plots and logs will be saved and logged to W&B under the specified experiment name.

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

For more details refer to [documentation](docs/index.md).
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

MIT License (see [LICENSE](LICENSE))
