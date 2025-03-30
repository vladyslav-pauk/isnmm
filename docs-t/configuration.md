## `config`

Below is an example of experiment configuration for a single training run:
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
To initiate a sweep, provide a list of values to scan, e.g. to compare NISCA and CNAE models, set `model_name: [nisca, cnae]`.