# Installation

## Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.11
- CUDA (optional but recommended)
- Weights & Biases

## Setup

```bash
git clone https://github.com/vladyslav-pauk/isnmm.git
cd isnmm
pip install -r requirements.txt
```

### Docker (recommended)

```bash
docker build -t nisca .
docker run --gpus all -it --rm -v $PWD:/app nisca
```
