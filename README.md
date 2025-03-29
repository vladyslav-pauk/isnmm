# Deep Generative Modeling for Hyperspectral & Medical Imaging

## 🧠 Overview

This repository implements **NISCA** — a modular deep generative framework for unsupervised source separation from high-dimensional imaging data such as **hyperspectral satellite images** and **dynamic contrast-enhanced MRI scans**. It leverages **variational autoencoders (VAEs)** with **geometric constraints** (e.g., simplex priors) and **post-nonlinear decoders** to recover interpretable latent factors under **nonlinear** and **noisy** mixing.

This work is part of the master's thesis by Vladyslav Pauk under the supervision of Prof. Xiao Fu at Oregon State University.

---

## 📁 Repository Structure

```
.
├── experiments/              # Experiment folders with data/model config JSON files
│   └── {experiment}/
│       └── config/
│           ├── data.json
│           ├── model/
│           │   └── {model_name}.json
│           └── sweep/
│               └── {sweep_config}.json
├── model/                    # Main model definitions: NISCA, CNAE, VASCA, etc.
│   ├── architectures/        # Encoder-decoder variants
│   ├── config/               # Default model configuration JSONs
│   ├── modules/              # LightningModule wrappers for training
│   ├── benchmarks/           # MVES and other classic baselines
│   └── metric/               # Evaluation metrics (e.g., separation, identifiability)
├── modules/                  # Optimizer and transform libraries
│   ├── optimizer/            # Augmented Lagrangian optimizer
│   ├── network/              # CNN, FCN, KAN, positive linear layers
│   └── transform/            # Nonlinear transforms (logit, glogit, etc.)
├── data/                     # Data loaders: MRI, hyperspectral, synthetic, etc.
│   ├── {domain}.py           # MRI, hyperspectral, astronomical, EEG, etc.
│   ├── *.json                # Dataset configurations
│   └── plots/                # Diagnostic plots (MRI, tensor slices)
├── scripts/                  # High-level wrappers and analysis tools
│   ├── run_dataset.py
│   ├── run_sweep.py
│   ├── analyze_sweep.py
│   ├── explore_model.py
│   └── compare_models.py
├── helpers/                  # Training entrypoint and sweep orchestration
│   ├── trainer.py            # Core train_model() function
│   ├── generate_data.py      # Synthetic data simulation
│   └── sweep_runner.py       # Grid & random search over configs
├── utils/                    # Logging (W&B), metrics, plotting, config parsing
│   ├── wandb_tools.py
│   ├── plot_tools.py
│   └── config_tools.py
└── notebooks/                # Optional Jupyter notebooks for visualization
```

---

## 🚀 Getting Started

### Installation

> Requires Python 3.11+

```bash
pip install -r requirements.txt
```

Set up W&B credentials:
```bash
wandb login
```

---

## 🔧 Usage

### 1. Train a Model

```bash
python helpers/trainer.py --experiment simplex_recovery --model nisca
```

This reads configuration from:
- `experiments/simplex_recovery/config/data.json`
- `experiments/simplex_recovery/config/model/nisca.json`

Optional overrides (e.g., batch size):
```bash
python helpers/trainer.py --experiment simplex_recovery --model nisca --batch_size 256
```

---

### 2. Run a Sweep

```bash
python scripts/run_sweep.py --experiment simplex_recovery --sweep sweep/model-param.json
```

Sweep configs are stored under:
```
experiments/{experiment_name}/config/sweep/
```

---

### 3. Visualize Results

```bash
python scripts/analyze_sweep.py --experiment simplex_recovery
python scripts/explore_model.py --experiment simplex_recovery --model nisca
```

Plots and logs will be saved and logged to W&B under the specified experiment name.

---

## 📊 Datasets

The framework supports:

- ✅ Synthetic mixtures with known ground truth
- ✅ Hyperspectral satellite images (Urban, Cuprite, Samson)
- ✅ Public DCE-MRI volumes
- ✅ Financial and astronomical data

All datasets are configured using `data.json` with preprocessing and loading logic defined in `data/*.py`.

---

## 🧠 Models

Implemented models include:

- `nisca`: Nonlinear ICA with simplex prior
- `vasca`: Variational simplex component analysis
- `cnae`: Constrained nonlinear autoencoder
- `nica`: Nonlinear ICA baseline
- `snae`: Simplex autoencoder
- `aevb`: Standard VAE (baseline)

Each model has its own encoder/decoder class under `model/architectures` and a training logic module in `model/modules`.

---

## 📏 Metrics

The following metrics are supported:

- **Reconstruction error** (RMSE)
- **Subspace distance** (Amari distance, spectral angle)
- **Residual nonlinearity**
- **Simplex mismatch**
- **Constraint error**
- **PSNR** and **R²**
- **Separation & identifiability**

All metrics are computed via `model/metric/*.py` and logged to W&B.

---

## 🛠️ Technological Stack

- **PyTorch Lightning** for training
- **Weights & Biases (W&B)** for logging
- **NumPy**, **Matplotlib**, **Scikit-learn**
- **Docker** + **GCP** compatibility
- **Configurable JSON experiments**
- Optional CUDA acceleration

---

## 📄 Publication

- [**Master's Thesis**](https://github.com/vladyslav-pauk/isnmm/blob/master/docs/thesis.pdf): *Deep Generative Modeling for Hyperspectral & Medical Imaging*, Vladyslav Pauk, OSU (2024)
- **IEEE Preprint**: Coming soon, see draft [here](https://arxiv.org/abs/2401.00000)

---

## 📚 Citation

```bibtex
@misc{pauk2024generative,
  author = {Vladyslav Pauk},
  title = {Deep Generative Modeling for Hyperspectral \& Medical Imaging},
  year = {2024},
  note = {Master's Thesis, Oregon State University}
}
```

---

## 🤝 Contact

- **Author**: Vladyslav Pauk  
- **Email**: [vlad.paukv@oregonstate.edu](mailto:vlad.paukv@oregonstate.edu)  
- **Website**: [linkedin.com/in/vladyslav-pauk](https://linkedin.com/in/vladyslav-pauk)

---





# Deep Generative Modeling for Hyperspectral & Medical Imaging

## Overview

This repository presents codebase, datasets, and documentation for NISCA, a scalable deep generative model designed to recover interpretable latent representations from high-dimensional imaging data. The model supports unsupervised source separation under nonlinear, noisy, and high-dimensional conditions—a setting common in medical imaging (DCE-MRI) and remote sensing (hyperspectral satellite images). 

[//]: # (This repository contains the codebase, datasets, and documentation for the project **"Deep Generative Modeling for Hyperspectral & Medical Imaging"** by **Dr. Vladyslav Pauk**, in collaboration with **Prof. Xiao Fu** at **Oregon State University**.)

[//]: # (The project focuses on developing a deep generative model for unsupervised tissue and material separation from high-dimensional imaging data such as **hyperspectral satellite images** and **DCE-MRI scans**. The approach extends **VASCA** using deep architecture, achieving interpretable and identifiable representations of mixed sources under nonlinear and noisy conditions.)

[//]: # (The project focuses on developing a deep generative model for hyperspectral and medical imaging data, specifically for unsupervised tissue and material separation. The model is based on a variational autoencoder &#40;VAE&#41; framework with geometric constraints, allowing for the identification of latent sources in high-dimensional data from satellite images and DCE-MRI scans.)

[//]: # (A scalable, interpretable latent variable model for unsupervised tissue and material separation from high-dimensional imaging data, including hyperspectral satellite images and DCE-MRI scans. This work extends post-nonlinear ICA using variational autoencoders &#40;VAEs&#41; with geometric constraints.)


Key features:
- Bayesian inference via deep variational autoencoders
- Geometrically constrained latent space (e.g., simplex priors) suitable for categorical ground truth
- Post-nonlinear decoder architecture
- Theoretical identifiability under nonlinear mixing
- Synthetic and real-world datasets (hyperspectral, DCE-MRI)
- Comprehensive experiment tracking with Weights & Biases
- Efficient implementation compatible with high-performance computing using CUDA
- Modular and scalable PyTorch Lightning pipeline
- Integrated support for GCP/Docker


For more information, see the Master thesis manuscript, IEEE preprint, or slides.


---

## 📁 Contents

- [Code](#code)
- [Usage](#usage)
  - [Notebooks](#notebooks)
  - [Training module](#training-module)
  - [Scripts](#scripts)
  - [Experiments](#experiments)
- [Data](#data)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [Publication](#publication)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)
- [References](#references)
- [🛠️ Technological Stack](#️technological-stack)

---

## 💾 Code

The code implements the NISCA algorithm for training VAEs on imaging data using the PyTorch Lightning framework, as well benchmarking against linear baselines, and nonlinear benchmark. The code is modular and designed for scalability, allowing for easy experimentation with different model architectures and configurations.
It supports both hyperspectral and medical data formats, includes synthetic simulation pipelines, and uses structured experiment tracking.

The core codebase includes:
- Modular PyTorch Lightning modules for training, evaluation, and logging
- Configurable architecture for encoder, decoder, and latent space
- Experiment tracking via Weights & Biases (W&B)
- Dockerized environment for reproducibility

Root directory structure:
```
.
├── datasets/           # Data loading, preprocessing, simulation
├── docs/               # Documentation (Markdown)
├── experiments/        # Experiment config files (JSON)
├── notebooks/          # Analysis, visualization, diagnostics
├── src/                # Main source scripts and tools
```

```
### Experiment Structure

```text
.
├── experiments/                # Experiment configurations
│   ├── vision/
│   │   └── config/
│   │       ├── data.json
│   │       └── model/
│   │           └── aevb.json
│   ├── fin_portfolio_return/
│   │   ├── yahoo.json
│   │   └── nisca.json
│   └── simplex_recovery/
│       └── config/
│           ├── data.json
│           ├── sweep/
│           │   ├── model-param.json
│           │   ├── single_run.json
│           │   ├── model-snr.json
│           │   └── model-datasize.json
│           └── model/
│               ├── nisca.json
│               └── vasca.json
├── wandb/                     # Weights & Biases run logs
│   └── run-*/                 # Each run with logs, files, and metadata
│       ├── logs/
│       ├── files/
│       └── tmp/
```

Source code structure:
```
├── src/                       # Main source scripts and tools
│   ├── data_module.py
│   ├── metrics_module.py
│   ├── evaluate.py
│   ├── schedule.py
│   ├── SPA.py
│   └── test_prism.py
├── experiments/               # Experiment drivers
│   ├── hyperspectral.py
│   ├── mri.py
│   ├── synthetic.py
│   └── portfolio.py
├── utils/                     # Utilities (logging, config, matrix ops)
│   ├── config_tools.py
│   ├── wandb_tools.py
│   ├── matrix_tools.py
│   ├── plot_tools.py
│   └── utils.py
├── image_marker/              # MRI marker web tool (Flask-based)
│   ├── app.py
│   ├── marked_pixels.mat
│   └── templates/
│       └── index.html
├── scripts/                   # Scripts for sweeping, evaluation, etc.
│   ├── run_sweep.py
│   ├── analyze_sweep.py
│   ├── explore_model.py
│   ├── compare_models.py
│   └── abundance_classifier.py
├── model/                     # Model architectures and configs
│   ├── config/
│   │   ├── snae.json
│   │   ├── nisca.json
│   │   └── vasca.json
│   ├── architectures/
│   │   ├── aevb.py
│   │   ├── vasca.py
│   │   └── nisca.py
│   ├── modules/
│   │   ├── lightning.py
│   │   ├── ae.py
│   │   └── vae.py
│   └── benchmarks/
│       └── mves.py
├── modules/                   # Modular components (networks, metrics, optimizers)
│   ├── optimizer/
│   │   └── augmented_lagrange.py
│   ├── network/
│   │   ├── fcn_constructor.py
│   │   ├── linear_positive.py
│   │   └── vision.py
│   ├── metric/
│   │   ├── separation.py
│   │   ├── subspace_distance.py
│   │   └── residual_nonlinearity.py
│   ├── distribution/
│   │   ├── standard.py
│   │   ├── mixture_model.py
│   │   └── location_scale.py
│   ├── data/
│   │   ├── hyperspectral.py
│   │   ├── dce_mri.py
│   │   └── synthetic.py
│   └── transform/
│       ├── logit_transform.py
│       ├── glogit_transform.py
│       └── nonlinear_component_wise.py
├── helpers/                   # Sweep & training helpers
│   ├── generate_data.py
│   ├── sweep_analyzer.py
│   └── trainer.py
```

---

## Installation

### Requirements

...

## 🧪 Usage

### Notebooks

Explore interactive Jupyter notebooks covering a range of topics from training to model evaluation and visualization, as well as some experiments.
- Model training walkthrough
- Posterior sampling and latent space visualization
- Component-wise reconstruction
- Quantitative evaluation and benchmark comparisons

### Training Module
Run train.py to train a model with a specific config file.
```bash
python train.py --config experiments/{experiment_name}/model_config.json
```

### Scripts

Run a hyperparameter sweep:
```bash
python run_sweep.py --config experiments/{experiment_name}/sweep_config.json
```

Schedule multiple experiments:
```bash
python schedule.py --config experiments/{experiment_name}/model_config.json
```


### Experiments

Each experiment is configured via two JSON files located in experiments/{experiment_name}/:
- model_config.json — specifies encoder/decoder architecture, latent space priors, optimization parameters
- data_config.json — defines the dataset source (synthetic, medical, satellite), preprocessing, batch size, etc.

Example fields in model_config.json:

The `experiments/` directory contains JSON configuration files for models and datasets:
- `model.json`: model architecture, prior type, latent dimension, etc.
- `data.json`: dataset path, loader parameters
- `sweep.json`: hyperparameter search grid
```json
{
  "project": "isnmm",
  "experiment": "lmm",
  "model": "vasca",
  "run_id": "run_name"
}
```

---

## 📊 Data

The framework supports the following data sources:
- Synthetic datasets for simulation (ground-truth available)
- Public hyperspectral satellite images (Urban, Cuprite, Samson)
- Anonymized DCE-MRI scans for medical imaging from public datasets

Formats:
- `.npy`, `.h5` for hyperspectral
- `.nii`, `.nii.gz` for MRI

Preprocessing steps include normalization, masking, and spatial augmentation.

---

## 📊 Results

The model:
- Achieves **~20% improvement** in parameter estimation vs. ICA/NMF
- Trains **2× faster** than unconstrained baselines
- Recovers interpretable latent factors with **simplex constraints**
- Provides **theoretical identifiability guarantees** for nonlinear mixing

[//]: # (The model achieves:)

[//]: # (- ~20% improvement in latent parameter recovery over ICA/NMF)

[//]: # (- 2× faster training convergence with constrained latent space)

[//]: # (- Improved interpretability and class separability)

[//]: # (- Strong generalization to unseen imaging samples)


Key metrics:
- RMSE, R², ELBO
- Residual Mutual Information
- Amari Distance (for latent subspace recovery)

---

## 🙏 Acknowledgements

This rep...

## 📄 Publication

- **Master's Thesis**: "Deep Generative Modeling for Hyperspectral Imaging"
- **IEEE Preprint**: (Coming Soon)
- **Slides**: [PDF/Google Slides Link]

---

## 📚 Citation

If you use this code or method, please cite:

```bibtex
@misc{pauk2024generative,
  author = {Vlad Pauk},
  title = {Deep Generative Modeling for Hyperspectral \& Medical Imaging},
  year = {2024},
  note = {Master's Thesis, Oregon State University}
}
```

---

## 📝 License

MIT License (see `LICENSE` file)

---

## 👥 Contributing

Pull requests, feedback, and discussions are welcome. Please submit issues or suggestions via GitHub.

---

## 📬 Contact

- **Author**: Dr. Vladyslav Pauk
- **Email**: [vlad.paukv@oregonstate.edu](mailto:vlad.pauk@oregonstate.edu)
- **Website**: [https://linkedin.com/vladyslav-pauk](https://linkedin.com/vladyslav-pauk)

---

## 🔗 References

- Hyvärinen et al., “Nonlinear ICA using auxiliary variables,” AISTATS 2019
- Locatello et al., “Disentanglement Challenges,” ICML 2019
- Miao & Qi, “Spectral Unmixing from Hyperspectral Imagery,” IEEE TGRS 2007

---
