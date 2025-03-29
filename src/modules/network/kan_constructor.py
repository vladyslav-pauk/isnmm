import torch
import torch.nn as nn


class KANConstructor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_units=100,
                 hidden_activation=None,
                 output_activation=None,
                 weight_initialization=None,
                 **kwargs):
        """
        Constructor for Kolmogorov-Arnold Network (KAN).

        Args:
            input_dim (int): The dimension of the input space.
            output_dim (int): The dimension of the output space.
            num_units (int): The number of units in the hidden layer (width of the network).
            hidden_activation (str): Activation function for hidden layers.
            output_activation (str): Activation function for the output layer.
            weight_initialization (str): Initialization method for weights.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units = num_units

        # Activations
        self.hidden_activation = getattr(nn, hidden_activation)() if hidden_activation else nn.ReLU()
        self.output_activation = getattr(nn, output_activation)() if output_activation else nn.Identity()

        # Weight initialization
        self.init_weights = getattr(nn.init, weight_initialization) if weight_initialization else lambda x: x

        # Build the network
        self.phi_layers = self._build_phi_layers()
        self.summation_layer = self._build_summation_layer()

    def _build_phi_layers(self):
        """
        Build the individual transformation layers (phi_layers) for each input dimension.
        """
        phi_layers = nn.ModuleList([
            self._init_layer(nn.Linear(1, self.num_units, bias=True))
            for _ in range(self.input_dim)
        ])
        return phi_layers

    def _build_summation_layer(self):
        """
        Build the summation layer, which combines the outputs of the transformation layers.
        """
        return self._init_layer(nn.Linear(self.num_units, self.output_dim, bias=True))

    def _init_layer(self, layer):
        """
        Apply weight initialization to a layer if specified.
        """
        if self.init_weights:
            self.init_weights(layer.weight)
        return layer

    def forward(self, x):
        """
        Forward pass of the Kolmogorov-Arnold Network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Transform each input dimension separately
        phi_outputs = []
        for i in range(self.input_dim):
            # Extract the i-th input feature and reshape to (batch_size, 1)
            x_i = x[:, i].unsqueeze(-1)
            # Pass through the corresponding phi_layer and apply activation
            phi_output = self.hidden_activation(self.phi_layers[i](x_i))
            phi_outputs.append(phi_output)

        # Combine the transformed features using summation
        combined_output = torch.sum(torch.stack(phi_outputs, dim=0), dim=0)

        # Pass through the summation layer and apply output activation
        output = self.output_activation(self.summation_layer(combined_output))
        return output