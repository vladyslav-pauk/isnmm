- C - constrained
- I - identifiable
- I - invertible
- N - nonlinear
- S - simplex

# Non-linear Auto-Encoder (NAE)

## CNAE

Deterministic (noiseless) model Constrained Non-Linear Autoencoder:

$$
    \min_{\theta} \frac{1}{N} \sum_{i=1}^{N} \left\| f(q(x_i)) - x_i \right\|_2^2,\\
    \text{ subject to } \bm 1^\top \bm q(\bm x_i) - 1 = 0 
$$

Optimized using augmented Lagrangian multiplier method.

## SNAE (NAES)

Deterministic (noiseless) model Nonlinear Autoencoder on a Simplex (Reparametrized)

Same loss as CNAE, but instead of the constraint we use mapping onto simplex (reparameterization).

# Logistic Variational Auto-Encoder (VASCA)

## VASCA

## NISCA

Add nonlinear layers to the decoder in VASCA.

## INISCA

Invertible NISCA with PNL encoder.

# Log-ratio Variational Auto-Encoder (NIVA)

## NIVA

Aitchison distribution Encoder

## CNIVA

Constrained Non-Linear Identifiable Variational Autoencoder

## INIVA

Variational Auto-Encoder on a Convex Hull (CHVAE)