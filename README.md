# NISCA

NISCA is a deep generative modeling framework for unsupervised latent source separation in spectral imaging data, with applications in remote sensing, medical imaging, astrophysics, and finance.
It extends traditional variational autoencoder architecture with the domain-specific inductive bias, by using geometrically constrained simplex priors, enabling interpretable and identifiable latent representations under noisy nonlinear mixing.

A production-grade implementation is provided, along with exhaustive documentation, and demonstration notebooks.
The framework is designed for **scalability** and **modularity**, allowing for easy experimentation with different model architectures and configurations.
It supports both hyperspectral and medical data formats, includes synthetic simulation pipelines, and uses structured experiment tracking.

[//]: # (geometric priors, e.g. Dirichlet, and nonlinear decoders)
[//]: # (complex post-nonlinear mixtures)
[//]: # (tissue and material separation from high-dimensional imaging data)

For more information, see Master thesis [manuscript](docs/thesis.pdf), IEEE [preprint](docs/preprint.pdf), and [slides](docs/slides.pdf), or jump to:

- [Highlights](#highlights)

- [Installation](#installation)

- [Usage](#usage)

- [Implementation](#implementation)

- [Publication](#publication)

- [Contact](#contact)

- [Contributing](#contributing)

- [License](#license)


## Highlights

### Model:
- **Bayesian** inference via deep variational autoencoders
- Geometrically constrained latent space (simplex priors) suitable for **categorical** ground truth
- Trainable **post-nonlinear decoder** supporting arbitrary invertible transforms
- Theoretical **identifiability** under nonlinear mixing and noise

### Implementation:
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

### Results

- Achieves **~20% improvement** in latent factor estimation over nICA/NMF
- Trains **2× faster** than constrained benchmarks
- Improved **interpretability** and class **separability**
- Provides **theoretical identifiability guarantees** for nonlinear mixing

[//]: # (The model achieves:)
[//]: # (- 2× faster training convergence with constrained latent space)
[//]: # (- Recovers **interpretable** latent factors)
[//]: # (- Strong generalization to unseen imaging samples)


## Installation

### Local (Virtual Environment)

Set up the project using a Python virtual environment.

1. Clone the repository by running:

```bash
  git clone https://github.com/vladyslav-pauk/nisca.git
```

2. Create and activate a virtual environment:
```bash
  python -m venv py-venv
  source py-venv/bin/activate  # Use `.\py-venv\Scripts\activate` on Windows
```

3. Install dependencies:
```bash
  pip install -r requirements.txt
```

4. Set up W&B credentials:
```bash
  wandb login
```

### Docker Container

Build and run the project in an isolated Docker environment.

1. Build the Docker image:
```bash
  docker build -t nisca
```

2. Run an interactive container with volume mounting:
```bash
  docker run -it --rm --name nisca-container \
    -v $(pwd):/app \
    nisca /bin/bash
```

3. Navigate to the project directory and launch training:
```bash
  cd /app
  PYTHONPATH=./ python src/scripts/run_sweep.py --experiment synthetic --sweep test_run
```

## Usage

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


### Experiments

The core functionality of this codebase revolves around executing **experiments**, which are defined through structured configuration files and can be launched from either the **command line** or within **Jupyter notebooks**.
It uses a grid method with a single deterministic run, suitable for benchmarking and ablation studies under controlled conditions. This setup is ideal for:

[//]: # (Each experiment encodes a specific computational task, typically including:)

- Multiple independent runs for model selection and statistical performance estimation
- Systematic hyperparameter tuning with grid or random search
- Controlled performance benchmarking and generalization analysis across different datasets
- Model selection and ablation studies across different network architectures, size, or training regimes
- Evaluating identifiability and latent factors recovery across different dataset sizes, or noise regimes

[//]: # (- Model training and validation )
[//]: # (- Hyperparameter sweep for model selection  )
[//]: # (- Comparative analysis of architecture or training regimes  )
[//]: # (- Ablation studies under controlled conditions)

This modular and configuration-driven design ensures reproducibility, scalability, and ease of integration into automated pipelines.

[//]: # (Each experiment is configured via `yaml` located in `experiments/{experiment_name}/config/` directory.)

Experiments are defined using `.yaml` or `.json` files located in the `experiments/` directory. Each configuration specifies model architecture, optimizer settings, and dataset parameters, enabling reproducible and scalable benchmarking.

Below is an example of a grid search configuration for training the NISCA model on the Pavia University hyperspectral dataset:


```yaml
method: grid                        # Sweep strategy
metric:                             # Objective to optimize
  goal: minimize
  name: validation_loss
parameters:
  experiment_name: hyperspectral
  data_model: PaviaU
  nonlinearity: [cnae]
  model_name: [nisca]
  trainer.max_epochs: [2000]
  early_stopping.min_delta: [1e-3]
  data_loader.batch_size: [100]
  dataset_size: [1000]
  observed_dim: [16]
  model.latent_dim: [null]
  snr: [25]
  model.sigma: [null]
  decoder.hidden_layers: 
    - h1: 128
  encoder.hidden_layers:
    - h1: 128
      h2: 64
      h3: 32
      h4: 16
  optimizer.lr.encoder: [0.001]
  optimizer.lr.decoder: [0.01]
  model.mc_samples: [1]
  torch_seed: [1]
  data_seed: [12]
```

[//]: # (paralelize)
[//]: # ( The main entry point is `src/scripts/run_sweep.py`, which orchestrates the training and evaluation of models based on the provided configurations.)
[//]: # (The framework supports multiple experiments, each with its own configuration.)
[//]: # (- model_config.json — specifies encoder/decoder architecture, latent space priors, optimization parameters)
[//]: # (- data_config.json — defines the dataset source &#40;synthetic, medical, satellite&#41;, preprocessing, batch size, etc.)
[//]: # (Example fields in model_config.json:)
[//]: # (The `experiments/` directory contains JSON configuration files for models and datasets:)
[//]: # (- `model.json`: model architecture, prior type, latent dimension, etc.)
[//]: # (- `data.json`: dataset path, loader parameters)
[//]: # (- `sweep.json`: hyperparameter search grid)
[//]: # (```json)
[//]: # ({)
[//]: # (  "project": "isnmm",)
[//]: # (  "experiment": "lmm",)
[//]: # (  "model": "vasca",)
[//]: # (  "run_id": "run_name")
[//]: # (})
[//]: # (```)

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

## Implementation

[//]: # (### Requirements)
The code implements the NISCA algorithm for training VAEs on imaging data using the PyTorch Lightning framework, as well benchmarking against linear baselines, and nonlinear benchmark. The code is modular and designed for scalability, allowing for easy experimentation with different model architectures and configurations.
It supports both hyperspectral and medical data formats, includes synthetic simulation pipelines, and uses structured experiment tracking.

The core codebase includes:
- Modular PyTorch Lightning modules for training, evaluation, and logging
- Configurable architecture for encoder, decoder, and latent space
- Experiment tracking via Weights & Biases (W&B)
- Dockerized environment for reproducibility

### Models

Implemented models include:

- `nisca`: Probabilistic nonlinear ICA with simplex prior.
- `vasca`: Variational simplex component analysis, linear baseline
- `cnae`: Constrained nonlinear autoencoder, nonlinear ICA benchmark
- `snae`: Simplex-projected autoencoder
- `aevb`: Standard VAE (baseline)

Each model has its own encoder/decoder class under `model/architectures` and a training logic module in `model/modules`.

### Metrics

The following metrics are supported:

- **Reconstruction error** (RMSE)
- **Subspace distance** (Amari distance, spectral angle)
- **Residual nonlinearity**
- **Simplex mismatch**
- **Constraint error**
- **PSNR** and **R²**
- **Separation & identifiability**

All metrics are computed via `model/metric/*.py` and logged to W&B.

### Datasets

The framework supports various datasets, including:

- Synthetic mixtures with known ground truth
- Hyperspectral satellite images (Urban, Cuprite, Samson)
- Public DCE-MRI volumes
- Financial and astronomical data

Datasets are loaded via `data/*.py` and can be configured in `data/*.json` files. The data loaders support formats:
- `.npy`, `.h5` for hyperspectral
- `.nii`, `.nii.gz` for MRI

Preprocessing steps include normalization, masking, and spatial augmentation.

### Repository Directory

The root directory contains:

```
.
├── datasets/                 # Datasets for training and evaluation
├── docs/                     # Documentation and publication materials
├── experiments/              # Experiment configuration files and logs
├── notebooks/                # Analysis, visualization, diagnostics
└── src/                      # Main source scripts and tools
```

Experiment folders contain JSON configuration files, logs, model checkpoints, and run results:

```
.
├── checkpoints/             # Model checkpoints named by run ID
├── config/                  # Experiment configuration files
│  └── test_run.yaml         # Experiment configuration
├── predictions/             # Model predictions
├── results/                 # Experiment results
├── wandb/                   # Weights & Biases run logs
└── info.md                  # Experiment metadata
```

[//]: # ()
[//]: # (Source code is organized as follows:)

[//]: # ()
[//]: # (```)

[//]: # (.)

[//]: # (├── experiments/              # Experiment drivers)

[//]: # (│   ├── hyperspectral.py)

[//]: # (│   ├── mri.py)

[//]: # (│   ├── synthetic.py)

[//]: # (│   └── portfolio.py)

[//]: # (├── model/                    # Main model definitions: NISCA, CNAE, VASCA, etc.)

[//]: # (│   ├── architectures/        # Encoder-decoder variants)

[//]: # (│   ├── config/               # Default model configuration JSONs)

[//]: # (│   ├── modules/              # LightningModule wrappers for training)

[//]: # (│   │   ├── lightning.py)

[//]: # (│   │   ├── ae.py)

[//]: # (│   │   └── vae.py)

[//]: # (│   └── benchmarks/           # MVES and other classic baselines)

[//]: # (├── modules/                  # Optimizer and transform libraries)

[//]: # (│   ├── optimizer/            # Augmented Lagrangian optimizer)

[//]: # (│   ├── network/              # CNN, FCN, KAN, positive linear layers)

[//]: # (│   ├── metric/               # Evaluation metrics &#40;e.g., separation, identifiability&#41;)

[//]: # (│   ├──transform/             # Nonlinear transforms &#40;logit, glogit, etc.&#41;)

[//]: # (│   ├── distribution/         # Distribution classes &#40;Dirichlet, Logistic-Normal&#41;)

[//]: # (│   └── data/                 # Data augmentation and preprocessing)

[//]: # (├── data/                     # Data loaders: MRI, hyperspectral, synthetic, etc.)

[//]: # (│   ├── {domain}.py           # MRI, hyperspectral, astronomical, EEG, etc.)

[//]: # (│   ├── *.json                # Dataset configurations)

[//]: # (│   └── plots/                # Diagnostic plots &#40;MRI, tensor slices&#41;)

[//]: # (├── scripts/                  # High-level wrappers and analysis tools)

[//]: # (│   ├── run_sweep.py          # Hyperparameter sweep entrypoint)

[//]: # (│   ├── analyze_sweep.py      # Sweep analysis and visualization)

[//]: # (│   ├── explore_model.py      # Model exploration and visualization)

[//]: # (│   ├── compare_models.py     # Compare models and visualize results)

[//]: # (│   └── abundance_classifier.py # Classifier for abundance maps)

[//]: # (├── helpers/                  # Training entrypoint and sweep orchestration)

[//]: # (│   ├── trainer.py            # Core train_model&#40;&#41; function)

[//]: # (│   ├── generate_data.py      # Synthetic data simulation)

[//]: # (│   ├── run_analyzer.py       # Analyze run results)

[//]: # (│   ├── sweep_analyzer.py     # Analyze sweep results)

[//]: # (│   └── sweep_runner.py       # Grid & random search over configs)

[//]: # (├── utils/                    # Logging &#40;W&B&#41;, metrics, plotting, config parsing)

[//]: # (│   ├── config_tools.py)

[//]: # (│   ├── wandb_tools.py)

[//]: # (│   ├── matrix_tools.py)

[//]: # (│   ├── plot_tools.py)

[//]: # (│   └── utils.py)

[//]: # (└── notebooks/                # Optional Jupyter notebooks for visualization)

[//]: # (```)

## Publication

- [**Master's Thesis**](https://github.com/vladyslav-pauk/isnmm/blob/master/docs/thesis.pdf): *Deep Generative Modeling for Hyperspectral & Medical Imaging*, Vladyslav Pauk, OSU (2024).
- **IEEE Preprint**: Coming soon, see draft [here](https://github.com/vladyslav-pauk/isnmm/blob/master/docs/preprint.pdf).

[//]: # (If you use this code or method, please cite )
BibTeX citation:

```bibtex
@misc{pauk2024generative,
  author = {Vladyslav Pauk},
  title = {Post-Nonlinear Mixture Identification via Variational Auto-Encoding},
  year = {2024},
  note = {Master's Thesis, Oregon State University}
}
```

[//]: # (## Acknowledgements)

[//]: # (This work is part of the master's thesis by Vladyslav Pauk under the supervision of Prof. Xiao Fu at Oregon State University.)
[//]: # ( Vladyslav Pauk**, in collaboration with **Prof. Xiao Fu** at **Oregon State University**.)


## Contact

- **Author**: Vladyslav Pauk  
- **Email**: [paukvp@gmail.com](mailto:paukvp@gmail.com)  
- **Website**: [linkedin.com/vladyslav-pauk](https://www.linkedin.com/in/vladyslav-pauk)


## Contributing

Pull requests, feedback, and discussions are welcome. Please submit issues or suggestions via GitHub.


## License

MIT License (see [LICENSE](LICENSE))
