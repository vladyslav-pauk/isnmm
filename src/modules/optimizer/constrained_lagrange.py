import torch
from torch.optim import Adam, Optimizer


class ConstrainedLagrangeOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, rho=1e2, inner_iters=1, n_sample=100, input_dim=10):
        """
        A custom optimizer that performs Adam gradient updates and also updates custom
        buffers such as Lagrange multipliers and F_buffer.

        Args:
            params: Model parameters to optimize.
            lr: Learning rate.
            rho: Coefficient for updating Lagrange multipliers.
            inner_iters: Number of iterations between multiplier updates.
            n_sample: Number of samples for buffer size.
            input_dim: Dimensionality of input data.
        """
        # Define the optimizer's default hyperparameters
        defaults = dict(lr=lr, rho=rho, inner_iters=inner_iters)

        # Initialize the Adam optimizer
        self.base_optimizer = Adam(params, lr=lr)

        # Initialize custom buffers
        self.F_buffer = torch.zeros((n_sample, input_dim))  # Example buffer
        self.count_buffer = torch.zeros(n_sample, dtype=torch.int32)
        self.mult = torch.zeros(n_sample)  # Lagrange multipliers

        # Keep track of internal state for updates
        self.n_sample = n_sample
        self.input_dim = input_dim
        self.global_step = 0

        super(ConstrainedLagrangeOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step (parameter update) and also update custom buffers.

        Args:
            closure: A closure that re-evaluates the model and returns the loss.
        """
        # Call the closure if provided
        loss = None
        if closure is not None:
            loss = closure()

        # Use the Adam optimizer's step for parameter updates
        self.base_optimizer.step()

        # Custom logic for updating Lagrange multipliers and buffers
        self.update_multipliers()

        # Increment the global step
        self.global_step += 1

        return loss

    def update_multipliers(self):
        """
        Custom logic to update Lagrange multipliers and reset buffers.
        This is run every 'inner_iters' steps.
        """
        # Check if we need to update Lagrange multipliers
        if (self.global_step + 1) % self.defaults['inner_iters'] == 0:
            idxes = self.count_buffer.nonzero(as_tuple=True)[0]
            F = self.F_buffer[idxes]
            diff = torch.sum(F, dim=1) - 1.0

            # Update Lagrange multipliers
            self.mult[idxes] += self.defaults['rho'] * diff

            # Reset the buffers
            self.F_buffer[idxes] = 0.0
            self.count_buffer[idxes] = 0

    def zero_grad(self):
        """
        Reset gradients of all model parameters (like in standard optimizers).
        """
        self.base_optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        """
        Load optimizer state including the base Adam optimizer's state.
        """
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        super(ConstrainedLagrangeOptimizer, self).load_state_dict(state_dict['self'])

    def state_dict(self):
        """
        Return the state of the optimizer.
        """
        state = super(ConstrainedLagrangeOptimizer, self).state_dict()
        state['base_optimizer'] = self.base_optimizer.state_dict()
        return state