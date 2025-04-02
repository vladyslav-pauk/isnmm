
[//]: # (## Prerequisites)

[//]: # ()
[//]: # (- Python ≥ 3.8)

[//]: # (- PyTorch ≥ 1.11)

[//]: # (- CUDA &#40;optional but recommended&#41;)

[//]: # (- Weights & Biases)

[//]: # ()
[//]: # (## Setup)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone https://github.com/vladyslav-pauk/isnmm.git)

[//]: # (cd isnmm)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

[//]: # ()
[//]: # (### Docker &#40;recommended&#41;)

[//]: # ()
[//]: # (```bash)

[//]: # (docker build -t nisca .)

[//]: # (docker run --gpus all -it --rm -v $PWD:/app nisca)

[//]: # (```)


# Installation Guide

## Local Installation

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

3. Install [dependencies](requirements.txt):
```bash
  pip install -r requirements.txt
```

4. Set up W&B credentials:
```bash
  wandb login
```

## Cloud Deployment

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

## CUDA Support

To enable CUDA support, set the environment variable `CUDA_VISIBLE_DEVICES` to the desired GPU ID(s):

```bash
  export CUDA_VISIBLE_DEVICES=0,1
```

You can run them in a Jupyter environment or convert them to scripts using `nbconvert`.

```bash
  jupyter nbconvert --to script notebook.ipynb
```