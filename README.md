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

Code structure:
```
.
├── models/             # VAE architectures (encoder, decoder, priors)
├── data/               # Data loading, preprocessing, simulation
├── experiments/        # Experiment config files (JSON)
├── utils/              # Training, logging, metric utilities
├── train.py            # Main training entrypoint
├── run_sweep.py        # Sweep launcher
├── schedule.py         # Scheduling training jobs
└── notebooks/          # Analysis, visualization, diagnostics
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

- **Author**: Vlad Pauk
- **Email**: [vlad.pauk@oregonstate.edu](mailto:vlad.pauk@oregonstate.edu)
- **Website**: [https://your-portfolio.com](https://your-portfolio.com) *(replace with actual)*

---

## 🔗 References

- Hyvärinen et al., “Nonlinear ICA using auxiliary variables,” AISTATS 2019
- Locatello et al., “Disentanglement Challenges,” ICML 2019
- Miao & Qi, “Spectral Unmixing from Hyperspectral Imagery,” IEEE TGRS 2007

---

## 🛠️ Technological Stack
