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
