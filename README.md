# NISCA

NISCA is a deep generative modeling framework for unsupervised latent source separation in spectral imaging data, with applications in remote sensing, medical imaging, astrophysics, and finance.
It extends traditional variational autoencoder architecture with the domain-specific inductive bias, by using geometrically constrained simplex priors, enabling interpretable and identifiable latent representations under noisy nonlinear mixing.
A scalable production-grade implementation is provided, along with exhaustive documentation, and demonstration notebooks.
The project supports datasets from hyperspectral satellite images, dynamic contrast-enhanced (DCE) MRI scans, and synthetic simulations.

[//]: # (geometric priors &#40;e.g., Dirichlet&#41; and nonlinear decoders, 
 even in complex post-nonlinear mixtures.)
[//]: # ( tissue and material separation from high-dimensional imaging data, including hyperspectral satellite images and dynamic contrast-enhanced &#40;DCE&#41; MRI scans)

### Highlights

[//]: # (- Bayesian inference via deep variational autoencoders)

[//]: # (- Geometrically constrained latent space &#40;e.g., simplex priors&#41; suitable for categorical ground truth)

[//]: # (- Post-nonlinear decoder architecture)

[//]: # (- Theoretical identifiability under nonlinear mixing)

[//]: # (- Synthetic and real-world datasets &#40;hyperspectral, DCE-MRI&#41;)

[//]: # (- Comprehensive experiment tracking with Weights & Biases)

[//]: # (- Efficient implementation compatible with high-performance computing using CUDA)

[//]: # (- Modular and scalable PyTorch Lightning pipeline)

[//]: # (- Integrated support for GCP/Docker)

[//]: # (The code implements the NISCA algorithm for training VAEs on imaging data using the PyTorch Lightning framework, as well benchmarking against linear baselines, and nonlinear benchmark. The code is modular and designed for scalability, allowing for easy experimentation with different model architectures and configurations.)

[//]: # (It supports both hyperspectral and medical data formats, includes synthetic simulation pipelines, and uses structured experiment tracking.)

[//]: # ()
[//]: # (The core codebase includes:)

[//]: # (- Modular PyTorch Lightning modules for training, evaluation, and logging)

[//]: # (- Configurable architecture for encoder, decoder, and latent space)

[//]: # (- Experiment tracking via Weights & Biases &#40;W&B&#41;)

[//]: # (- Dockerized environment for reproducibility)


- **Variational autoencoders** with simplex-constrained latent priors (Dirichlet, Logistic-Normal)
- **Post-nonlinear decoder** supporting arbitrary invertible transforms
- **Synthetic + real-world** data support (Urban, Cuprite, MRI, financial datasets)
- **Augmented Lagrangian optimization** for constrained training
- **Multi-experiment orchestration**, WandB logging, sweeping, and GCP/Docker compatibility
- **Metrics for identifiability and recovery**: subspace distance, Amari index, mutual info, etc.

### Technological Stack

- **PyTorch Lightning** for training and evaluation
- **Weights & Biases (W&B)** for logging
- **NumPy**, **Matplotlib**, **Scikit-learn**
- **Docker** + **GCP** compatibility
- **Configurable JSON experiments**
- Optional **CUDA** acceleration

For more information, see Master thesis [manuscript](docs/thesis.pdf), IEEE [preprint](docs/preprint.pdf), and [slides](docs/slides.pdf), or jump to:


[//]: # (- [Highlights]&#40;#highlights&#41;)

- [Installation](#installation)

- [Usage](#usage)

- [Codebase](#codebase)

- [Publication](#publication)

- [Contact](#contact)

- [Contributing](#contributing)

- [License](#license)

## Installation

To clone the repository, run:

```bash
  git clone https://github.com/vladyslav-pauk/nisca.git
```
Install dependencies:
```bash
  pip install -r requirements.txt
```

Set up W&B credentials:
```bash
  wandb login
```


## Usage

### Train a Model

To train a model, run:


```bash
  PYTHONPATH=./ python src/scripts/run_sweep.py --experiment synthetic --sweep test_run
```

This reads configuration from:
- `experiments/simplex_recovery/config/data.json`
- `experiments/simplex_recovery/config/model/nisca.json`

Job results are stored under `experiments/{experiment_name}/config/sweep/`

### Model Evaluation

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

### CUDA

To enable CUDA support, set the environment variable `CUDA_VISIBLE_DEVICES` to the desired GPU ID(s):

```bash
  export CUDA_VISIBLE_DEVICES=0,1
```

### Jupyter Notebooks
The repository includes Jupyter notebooks for interactive exploration of the models and datasets and reproducing results from the paper.

- [**Model Training**](notebooks/model_training.ipynb): Walkthrough of training a model with different configurations.
- [**Training Evaluation**](notebooks/quantitative_evaluation.ipynb): Benchmarking against classic methods and analyzing results.
- [**Synthetic Data**](notebooks/mixture_model.ipynb): Generating and visualizing synthetic data for testing.

You can run them in a Jupyter environment or convert them to scripts using `nbconvert`.

```bash
  jupyter nbconvert --to script notebook.ipynb
```

### Containerization


## Codebase


### Datasets

The framework supports:

- Synthetic mixtures with known ground truth
- Hyperspectral satellite images (Urban, Cuprite, Samson)
- Public DCE-MRI volumes
- Financial and astronomical data

All datasets are configured using `data.json` with preprocessing and loading logic defined in `data/*.py`.

### Models

Implemented models include:

- `nisca`: Nonlinear ICA with simplex prior
- `vasca`: Variational simplex component analysis
- `cnae`: Constrained nonlinear autoencoder
- `nica`: Nonlinear ICA baseline
- `snae`: Simplex autoencoder
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


### Directory Structure

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



## Publication

- [**Master's Thesis**](https://github.com/vladyslav-pauk/isnmm/blob/master/docs/thesis.pdf): *Deep Generative Modeling for Hyperspectral & Medical Imaging*, Vladyslav Pauk, OSU (2024).
- **IEEE Preprint**: Coming soon, see draft [here](https://github.com/vladyslav-pauk/isnmm/blob/master/docs/preprint.pdf).

BibTeX citation:

```bibtex
@misc{pauk2024generative,
  author = {Vladyslav Pauk},
  title = {Post-Nonlinear Mixture Identification via Variational Auto-Encoding},
  year = {2024},
  note = {Master's Thesis, Oregon State University}
}
```


## Contact

- **Author**: Vladyslav Pauk  
- **Email**: [paukvp@gmail.com](mailto:paukvp@gmail.com)  
- **Website**: [linkedin.com/vladyslav-pauk](https://www.linkedin.com/in/vladyslav-pauk)


## Contributing

Pull requests, feedback, and discussions are welcome. Please submit issues or suggestions via GitHub.


## License

MIT License (see [LICENSE](LICENSE))



[//]: # ()
[//]: # (Root directory structure:)

[//]: # (```)

[//]: # (.)

[//]: # (├── datasets/           # Data loading, preprocessing, simulation)

[//]: # (├── docs/               # Documentation &#40;Markdown&#41;)

[//]: # (├── experiments/        # Experiment config files &#40;JSON&#41;)

[//]: # (├── notebooks/          # Analysis, visualization, diagnostics)

[//]: # (├── src/                # Main source scripts and tools)

[//]: # (```)

[//]: # ()
[//]: # (```)

[//]: # (### Experiment Structure)

[//]: # ()
[//]: # (```text)

[//]: # (.)

[//]: # (├── experiments/                # Experiment configurations)

[//]: # (│   ├── vision/)

[//]: # (│   │   └── config/)

[//]: # (│   │       ├── data.json)

[//]: # (│   │       └── model/)

[//]: # (│   │           └── aevb.json)

[//]: # (│   ├── fin_portfolio_return/)

[//]: # (│   │   ├── yahoo.json)

[//]: # (│   │   └── nisca.json)

[//]: # (│   └── simplex_recovery/)

[//]: # (│       └── config/)

[//]: # (│           ├── data.json)

[//]: # (│           ├── sweep/)

[//]: # (│           │   ├── model-param.json)

[//]: # (│           │   ├── single_run.json)

[//]: # (│           │   ├── model-snr.json)

[//]: # (│           │   └── model-datasize.json)

[//]: # (│           └── model/)

[//]: # (│               ├── nisca.json)

[//]: # (│               └── vasca.json)

[//]: # (├── wandb/                     # Weights & Biases run logs)

[//]: # (│   └── run-*/                 # Each run with logs, files, and metadata)

[//]: # (│       ├── logs/)

[//]: # (│       ├── files/)

[//]: # (│       └── tmp/)

[//]: # (```)

[//]: # ()
[//]: # (Source code structure:)

[//]: # (```)

[//]: # (├── src/                       # Main source scripts and tools)

[//]: # (│   ├── data_module.py)

[//]: # (│   ├── metrics_module.py)

[//]: # (│   ├── evaluate.py)

[//]: # (│   ├── schedule.py)

[//]: # (│   ├── SPA.py)

[//]: # (│   └── test_prism.py)

[//]: # (├── experiments/               # Experiment drivers)

[//]: # (│   ├── hyperspectral.py)

[//]: # (│   ├── mri.py)

[//]: # (│   ├── synthetic.py)

[//]: # (│   └── portfolio.py)

[//]: # (├── utils/                     # Utilities &#40;logging, config, matrix ops&#41;)

[//]: # (│   ├── config_tools.py)

[//]: # (│   ├── wandb_tools.py)

[//]: # (│   ├── matrix_tools.py)

[//]: # (│   ├── plot_tools.py)

[//]: # (│   └── utils.py)

[//]: # (├── image_marker/              # MRI marker web tool &#40;Flask-based&#41;)

[//]: # (│   ├── app.py)

[//]: # (│   ├── marked_pixels.mat)

[//]: # (│   └── templates/)

[//]: # (│       └── index.html)

[//]: # (├── scripts/                   # Scripts for sweeping, evaluation, etc.)

[//]: # (│   ├── run_sweep.py)

[//]: # (│   ├── analyze_sweep.py)

[//]: # (│   ├── explore_model.py)

[//]: # (│   ├── compare_models.py)

[//]: # (│   └── abundance_classifier.py)

[//]: # (├── model/                     # Model architectures and configs)

[//]: # (│   ├── config/)

[//]: # (│   │   ├── snae.json)

[//]: # (│   │   ├── nisca.json)

[//]: # (│   │   └── vasca.json)

[//]: # (│   ├── architectures/)

[//]: # (│   │   ├── aevb.py)

[//]: # (│   │   ├── vasca.py)

[//]: # (│   │   └── nisca.py)

[//]: # (│   ├── modules/)

[//]: # (│   │   ├── lightning.py)

[//]: # (│   │   ├── ae.py)

[//]: # (│   │   └── vae.py)

[//]: # (│   └── benchmarks/)

[//]: # (│       └── mves.py)

[//]: # (├── modules/                   # Modular components &#40;networks, metrics, optimizers&#41;)

[//]: # (│   ├── optimizer/)

[//]: # (│   │   └── augmented_lagrange.py)

[//]: # (│   ├── network/)

[//]: # (│   │   ├── fcn_constructor.py)

[//]: # (│   │   ├── linear_positive.py)

[//]: # (│   │   └── vision.py)

[//]: # (│   ├── metric/)

[//]: # (│   │   ├── separation.py)

[//]: # (│   │   ├── subspace_distance.py)

[//]: # (│   │   └── residual_nonlinearity.py)

[//]: # (│   ├── distribution/)

[//]: # (│   │   ├── standard.py)

[//]: # (│   │   ├── mixture_model.py)

[//]: # (│   │   └── location_scale.py)

[//]: # (│   ├── data/)

[//]: # (│   │   ├── hyperspectral.py)

[//]: # (│   │   ├── dce_mri.py)

[//]: # (│   │   └── synthetic.py)

[//]: # (│   └── transform/)

[//]: # (│       ├── logit_transform.py)

[//]: # (│       ├── glogit_transform.py)

[//]: # (│       └── nonlinear_component_wise.py)

[//]: # (├── helpers/                   # Sweep & training helpers)

[//]: # (│   ├── generate_data.py)

[//]: # (│   ├── sweep_analyzer.py)

[//]: # (│   └── trainer.py)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (### Requirements)

[//]: # ()
[//]: # (...)

[//]: # ()
[//]: # (## 🧪 Usage)

[//]: # ()
[//]: # (### Notebooks)

[//]: # ()
[//]: # (Explore interactive Jupyter notebooks covering a range of topics from training to model evaluation and visualization, as well as some experiments.)

[//]: # (- Model training walkthrough)

[//]: # (- Posterior sampling and latent space visualization)

[//]: # (- Component-wise reconstruction)

[//]: # (- Quantitative evaluation and benchmark comparisons)

[//]: # ()
[//]: # (### Training Module)

[//]: # (Run train.py to train a model with a specific config file.)

[//]: # (```bash)

[//]: # (python train.py --config experiments/{experiment_name}/model_config.json)

[//]: # (```)

[//]: # ()
[//]: # (### Scripts)

[//]: # ()
[//]: # (Run a hyperparameter sweep:)

[//]: # (```bash)

[//]: # (python run_sweep.py --config experiments/{experiment_name}/sweep_config.json)

[//]: # (```)

[//]: # ()
[//]: # (Schedule multiple experiments:)

[//]: # (```bash)

[//]: # (python schedule.py --config experiments/{experiment_name}/model_config.json)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (### Experiments)

[//]: # ()
[//]: # (Each experiment is configured via two JSON files located in experiments/{experiment_name}/:)

[//]: # (- model_config.json — specifies encoder/decoder architecture, latent space priors, optimization parameters)

[//]: # (- data_config.json — defines the dataset source &#40;synthetic, medical, satellite&#41;, preprocessing, batch size, etc.)

[//]: # ()
[//]: # (Example fields in model_config.json:)

[//]: # ()
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

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 📊 Data)

[//]: # ()
[//]: # (The framework supports the following data sources:)

[//]: # (- Synthetic datasets for simulation &#40;ground-truth available&#41;)

[//]: # (- Public hyperspectral satellite images &#40;Urban, Cuprite, Samson&#41;)

[//]: # (- Anonymized DCE-MRI scans for medical imaging from public datasets)

[//]: # ()
[//]: # (Formats:)

[//]: # (- `.npy`, `.h5` for hyperspectral)

[//]: # (- `.nii`, `.nii.gz` for MRI)

[//]: # ()
[//]: # (Preprocessing steps include normalization, masking, and spatial augmentation.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 📊 Results)

[//]: # ()
[//]: # (The model:)

[//]: # (- Achieves **~20% improvement** in parameter estimation vs. ICA/NMF)

[//]: # (- Trains **2× faster** than unconstrained baselines)

[//]: # (- Recovers interpretable latent factors with **simplex constraints**)

[//]: # (- Provides **theoretical identifiability guarantees** for nonlinear mixing)

[//]: # ()
[//]: # ([//]: # &#40;The model achieves:&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- ~20% improvement in latent parameter recovery over ICA/NMF&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- 2× faster training convergence with constrained latent space&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- Improved interpretability and class separability&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- Strong generalization to unseen imaging samples&#41;)
[//]: # ()
[//]: # ()
[//]: # (Key metrics:)

[//]: # (- RMSE, R², ELBO)

[//]: # (- Residual Mutual Information)

[//]: # (- Amari Distance &#40;for latent subspace recovery&#41;)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 🙏 Acknowledgements)

[//]: # (This work is part of the master's thesis by Vladyslav Pauk under the supervision of Prof. Xiao Fu at Oregon State University.)
[//]: # ( Vladyslav Pauk**, in collaboration with **Prof. Xiao Fu** at **Oregon State University**.)

[//]: # ()
[//]: # (This rep...)

[//]: # ()
[//]: # (## 📄 Publication)

[//]: # ()
[//]: # (- **Master's Thesis**: "Deep Generative Modeling for Hyperspectral Imaging")

[//]: # (- **IEEE Preprint**: &#40;Coming Soon&#41;)

[//]: # (- **Slides**: [PDF/Google Slides Link])

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 📚 Citation)

[//]: # ()
[//]: # (If you use this code or method, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@misc{pauk2024generative,)

[//]: # (  author = {Vlad Pauk},)

[//]: # (  title = {Deep Generative Modeling for Hyperspectral \& Medical Imaging},)

[//]: # (  year = {2024},)

[//]: # (  note = {Master's Thesis, Oregon State University})

[//]: # (})

[//]: # (```)

[//]: # ()


[//]: # ()
[//]: # (## 📬 Contact)

[//]: # ()
[//]: # (- **Author**: Dr. Vladyslav Pauk)

[//]: # (- **Email**: [vlad.paukv@oregonstate.edu]&#40;mailto:vlad.pauk@oregonstate.edu&#41;)

[//]: # (- **Website**: [https://linkedin.com/vladyslav-pauk]&#40;https://linkedin.com/vladyslav-pauk&#41;)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 🔗 References)

[//]: # ()
[//]: # (- Hyvärinen et al., “Nonlinear ICA using auxiliary variables,” AISTATS 2019)

[//]: # (- Locatello et al., “Disentanglement Challenges,” ICML 2019)

[//]: # (- Miao & Qi, “Spectral Unmixing from Hyperspectral Imagery,” IEEE TGRS 2007)

[//]: # ()
[//]: # (---)
