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
- Stock market data

Datasets are loaded via `data/*.py` and can be configured in `data/*.json` files. The data loaders support formats:
- `.npy`, `.h5` for hyperspectral
- `.nii`, `.nii.gz` for MRI

Preprocessing steps include normalization, masking, and spatial augmentation.

### Repository Directory

The root directory contains:

[//]: # (```)

[//]: # (.)

[//]: # (├── datasets/                 # Datasets for training and evaluation)

[//]: # (├── docs/                     # Documentation and publication materials)

[//]: # (├── experiments/              # Experiment configuration files and logs)

[//]: # (├── notebooks/                # Analysis, visualization, diagnostics)

[//]: # (└── src/                      # Main source scripts and tools)

[//]: # (```)

```
.
├── README.md
├── notebooks/             # Jupyter notebooks for training, exploration, plotting
├── experiments/           # Experiment configs: model/data/sweeps (.json/.yaml)
├── docs/                  # Markdown and LaTeX documentation
├── src/                   # Source code
│   ├── model/             # Architectures, priors, benchmarks
│   ├── modules/           # Components: metrics, transforms, networks, distributions
│   ├── utils/             # Logging, plotting, config utils, WandB integration
│   ├── scripts/           # Main entrypoints: train, sweep, compare, analyze
│   ├── experiments/       # Task-specific experiment logic: hyperspectral, MRI, synthetic
│   ├── helpers/           # Scheduling, orchestration, plotting tools
│   ├── image_marker/      # Web app for ground truth pixel annotation
│   ├── data/               # Data loaders, preprocessing, augmentation
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

