import argparse
import sys
import torch
import scipy.io as sio
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scipy.linalg as spalg
import multiprocessing

# Set multiprocessing start method for MacOS
if sys.platform == 'darwin':
    multiprocessing.set_start_method('fork')

# Argument parser
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--s_dim", default=3, help="Dimensionality of s", type=int)
    parser.add_argument("--batch_size", default=1000, help="Batch size", type=int)
    parser.add_argument("--num_epochs", default=500, help="Number of epochs", type=int)
    parser.add_argument("--inner_iters", default=100, help="Update multipliers every N steps", type=int)
    parser.add_argument("--learning_rate", default=1e-3, help="Learning rate", type=float)
    parser.add_argument("--rho", default=1e2, help="Value of rho", type=float)
    parser.add_argument("--model_file_name", default='best_model_simplex.ckpt',
                        help="File name for best model saving", type=str)
    # Structure for encoder and decoder network
    parser.add_argument("--f_num_layers", default=3, help="Number of layers for f", type=int)
    parser.add_argument("--f_hidden_size", default=128, help="Number of hidden neurons for f", type=int)
    parser.add_argument("--q_num_layers", default=3, help="Number of layers for q", type=int)
    parser.add_argument("--q_hidden_size", default=128, help="Number of hidden neurons for q", type=int)

    return parser

# Dataset class
class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.data_len = data_tensor.shape[0]

    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return self.data_len

# The PNL unmixing model
class PNLModule(pl.LightningModule):
    def __init__(self, input_dim, f_size, q_size, s_dim, n_sample, rho, args, qs):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.qs = qs
        self.best_constraint_val = float('inf')
        self.subspace_dist_arr = []

        # Build the model
        self.input_dim = input_dim
        self.s_dim = s_dim
        self.n_sample = n_sample
        self.rho = rho

        # Multipliers
        self.mult = nn.Parameter(torch.randn(n_sample), requires_grad=False)

        # Encoding network
        self.e_net = self.build_network(self.input_dim, f_size, self.input_dim)

        # Decoding network
        self.d_net = self.build_network(self.input_dim, q_size, self.input_dim)

        # Buffers for multiplier updates
        self.register_buffer('F_buffer', torch.zeros((n_sample, input_dim)))
        self.register_buffer('count_buffer', torch.zeros(n_sample, dtype=torch.int32))

    def build_network(self, input_dim, hidden_sizes, output_dim):
        layers = []
        in_channels = input_dim
        for h in hidden_sizes:
            layers.append(nn.Conv1d(in_channels, h * input_dim, kernel_size=1, groups=input_dim))
            layers.append(nn.ReLU())
            in_channels = h * input_dim
        layers.append(nn.Conv1d(in_channels, output_dim, kernel_size=1, groups=input_dim))
        return nn.Sequential(*layers)

    def encode(self, x):
        y = self.e_net(x.unsqueeze(-1))
        return y.squeeze(-1)

    def decode(self, x):
        y = self.d_net(x.unsqueeze(-1))
        return y.squeeze(-1)

    def forward(self, x):
        fx = self.encode(x)
        qfx = self.decode(fx)
        return fx, qfx

    def training_step(self, batch, batch_idx):
        data, idxes = batch
        data = data.float()
        idxes = idxes.long()
        fx, qfx = self(data)
        # Compute losses
        total_loss, r_e, f_e, a_e = self.loss_function(fx, qfx, data, idxes)
        # Log losses
        # self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('reconstruction_error', r_e, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('feasible_error', f_e, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('augmented_error', a_e, on_step=True, on_epoch=True, prog_bar=True)
        # Store fx for multiplier update
        self.F_buffer[idxes] = fx.detach()
        self.count_buffer[idxes] += 1

        # Update multipliers every 'args.inner_iters' steps
        if (self.global_step + 1) % self.args.inner_iters == 0:
            self.update_multipliers()

        return total_loss

    def update_multipliers(self):
        # Only update multipliers for samples that have been seen
        idxes = self.count_buffer.nonzero(as_tuple=True)[0]
        F = self.F_buffer[idxes]
        diff = torch.sum(F, dim=1) - 1.0
        self.mult[idxes] += self.rho * diff
        squared_diff = torch.norm(diff) ** 2
        # self.log('squared_diff', squared_diff, prog_bar=True)

        # Save the model if the constraint value decreases
        if squared_diff < self.best_constraint_val:
            self.best_constraint_val = squared_diff
            print('\nSaving Model')
            self.trainer.save_checkpoint(self.args.model_file_name)

        # Compute the subspace distance
        # Move F to CPU if necessary
        F_cpu = F.cpu()
        qf, _ = torch.linalg.qr(F_cpu)
        subspace_dist = np.sin(spalg.subspace_angles(self.qs, qf.numpy()))[0]
        self.subspace_dist_arr.append(subspace_dist)
        self.log('subspace_distance', subspace_dist, prog_bar=True)

        # Reset buffers
        self.F_buffer[idxes] = 0.0
        self.count_buffer[idxes] = 0

    def loss_function(self, fx, qfx, x, idxes):
        loss_fn = nn.MSELoss(reduction='sum')
        tmp = torch.sum(fx, dim=1) - 1.0
        mult = self.mult[idxes]
        reconstruct_err = loss_fn(qfx, x) / x.shape[0]
        feasible_err = torch.dot(mult, tmp) / x.shape[0]
        augmented_err = torch.norm(tmp) ** 2 / x.shape[0]
        total_loss = reconstruct_err + feasible_err + self.rho / 2. * augmented_err
        return total_loss, reconstruct_err, feasible_err, augmented_err

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

# For better visualization
def visual_normalization(x):
    bound = 10
    x = x - np.amin(x)
    x = x / np.amax(x) * bound
    return x

# Evaluate the learned model
def evaluate(model, args, dataloader, device, mixture, x):
    # Load the best model
    checkpoint = torch.load(args.model_file_name, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # Forward
    with torch.no_grad():
        F = []
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device).float()
            fx = model.encode(data)
            F.append(fx.cpu().numpy())

        F = np.concatenate(F, axis=0)

    # Scatter plot the results
    for i in range(mixture.shape[1]):
        plt.subplot(1, mixture.shape[1], i + 1)
        # Plot the composition f ∘ g
        plt.scatter(mixture[:, i], visual_normalization(F[:, i]),
                    label=rf'$\hat{{f}}_{i+1}\circ g_{i+1}$', alpha=0.5)
        # Plot the generative function g
        plt.scatter(mixture[:, i], visual_normalization(x[:, i]), label=rf'$g_{i+1}$', alpha=0.5)

        plt.xlabel('input', fontsize=12)
        if i == 0:
            plt.ylabel('output', fontsize=12)

        plt.legend(fontsize=12)

    plt.show()

def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    torch.manual_seed(1)
    np.random.seed(12)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Read data file
    data_file = './post-nonlinear_simplex_synthetic_data.mat'
    data = sio.loadmat(data_file)

    # Read input
    x = data['x'].astype(np.float32)
    s_groundtruth = data['s']
    qs = data['s_q']
    mixture = data['linear_mixture']

    # Dimension of input
    n_sample = x.shape[0]
    n_feature = x.shape[1]

    # Build the PNL LightningModule
    model = PNLModule(input_dim=n_feature,
                      f_size=[args.f_hidden_size] * args.f_num_layers,
                      q_size=[args.q_hidden_size] * args.q_num_layers,
                      s_dim=args.s_dim,
                      n_sample=n_sample,
                      rho=args.rho,
                      args=args,
                      qs=qs)

    # Determine number of workers
    num_cores = multiprocessing.cpu_count()
    num_workers = num_cores // 2  # Adjust as needed
    print(f'Number of CPU cores: {num_cores}, using {num_workers} workers for DataLoader.')

    # Generate dataloader
    dataset = MyDataset(torch.from_numpy(x))
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    eval_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    # Set up Trainer
    max_epochs = args.num_epochs  # Keep the number of epochs as specified
    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator='auto',
                         devices=1 if torch.cuda.is_available() or torch.backends.mps.is_available() else None,
                         logger=False)

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader)

    # Evaluate the model and plot
    evaluate(model, args, eval_loader, device, mixture, x)

    # Print subspace distances
    print('Subspace distances:', model.subspace_dist_arr)

if __name__ == "__main__":
    main(sys.argv[1:])