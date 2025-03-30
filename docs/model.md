# Model Overview

## Problem Formulation

Given observations \(\mathbf{X} \in \mathbb{R}^{N \times D}\), we aim to recover latent components \(\mathbf{Z} \in \mathbb{R}^{N \times K}\) under a nonlinear mixture model:

\[
\mathbf{X} = f(\mathbf{Z}) + \varepsilon,\quad \mathbf{Z} \in \Delta^K
\]

Where:
- \(f: \Delta^K \rightarrow \mathbb{R}^D\) is a learnable, invertible decoder
- \(\Delta^K\) is the \(K\)-simplex
- \(\varepsilon\) denotes additive noise

## Architecture

- **Encoder**: Learns \(q_\phi(\mathbf{z}|\mathbf{x})\) via MLP
- **Latent Prior**: Dirichlet or Logistic-Normal
- **Decoder**: Nonlinear \(f_\theta(\mathbf{z})\), parameterized as MLP
- **Objective**:

\[
\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \mathrm{KL}[q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})]
\]

## Identifiability

NISCA provides theoretical guarantees of identifiability under:
- Simplex-constrained priors
- Sufficient nonlinearity and regularity of the decoder
- Finite SNR

The framework integrates subspace metrics, separation indices, and latent recovery benchmarks to evaluate identifiability in practice.
