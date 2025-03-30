# Configuration Guide

Experiments are defined via JSON/YAML files under the `experiments/` directory.

## Example Configuration (YAML)

```yaml
method: grid
parameters:
  model_name: [nisca]
  encoder.hidden_layers:
    - h1: 128
      h2: 64
  decoder.hidden_layers:
    - h1: 128
  optimizer.lr.encoder: [0.001]
  optimizer.lr.decoder: [0.01]
  trainer.max_epochs: [2000]
  snr: [25]
```

## Supported Keys

- `model_name`: One of `nisca`, `vasca`, `cnae`, `snae`, `aevb`
- `data_model`: Dataset identifier
- `latent_dim`, `snr`, `early_stopping.min_delta`
