# Experiment Design

## Structure

Each task is defined in `experiments/{task}/`:

- `config/`: Configuration YAML or JSON
- `results/`: Evaluation outputs
- `checkpoints/`: Model weights
- `wandb/`: Logging data

## Tasks

| Task          | Datasets           | Models           |
|---------------|--------------------|------------------|
| Synthetic     | Simulated mixtures | NISCA, VASCA     |
| Hyperspectral | Urban, Cuprite     | NISCA, CNAE      |
| MRI           | DCE-MRI            | NISCA, VASCA     |
| Finance       | Yahoo Finance      | NISCA            |
