import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# =========================
#        Data Loader
# =========================


class DataLoader1D(object):
    def __init__(self, x_data, y_data, nx=128, nt=100, sub=1, sub_t=1, new=True):
#         dataloader = MatReader(datapath)
        self.sub = sub
        self.sub_t = sub_t
        s = nx
        # if nx is odd
        if (s % 2) == 1:
            s = s - 1
        self.s = s // sub

        self.T = nt // sub_t
        self.new = new
        if new:
            self.T += 1
        self.x_data = x_data[:, 0:s:sub]
        self.y_data = y_data[:, 0:self.T:sub_t, 0:s:sub]

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        Xs = self.x_data[start:start + n_sample]
        ys = self.y_data[start:start + n_sample]

        if self.new:
            gridx = torch.tensor(np.linspace(0, 1, self.s + 1)[:-1], dtype=torch.float64)
            gridt = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float64)
        else:
            gridx = torch.tensor(np.linspace(0, 1, self.s), dtype=torch.float64)
            gridt = torch.tensor(np.linspace(0, 1, self.T + 1)[1:], dtype=torch.float64)
        gridx = gridx.reshape(1, 1, self.s)
        gridt = gridt.reshape(1, self.T, 1)

        Xs = Xs.reshape(n_sample, 1, self.s).repeat([1, self.T, 1])
        Xs = torch.stack([Xs, gridx.repeat([n_sample, self.T, 1]), gridt.repeat([n_sample, 1, self.s])], dim=3)
        dataset = torch.utils.data.TensorDataset(Xs, ys)
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

# =========================
#        Model Definition
# =========================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Spectral Convolution Layer using Fourier transforms.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes1 (int): Number of Fourier modes in the first dimension.
            modes2 (int): Number of Fourier modes in the second dimension.
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)

        # Initialize spectral weights
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        Forward pass for spectral convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width, stacked_terms).

        Returns:
            torch.Tensor: Output tensor after spectral convolution.
        """
        batchsize, in_channels, size1, size2 = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # Fourier transform
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batchsize, self.out_channels, size1, size2 // 2 + 1, x.shape[-1],
            device=x.device, dtype=torch.cfloat
        )

        # Apply spectral weights
        out_ft[:, :, :self.modes1, :self.modes2, :] += torch.einsum(
            "bixyt,ioxy->boxyt", x_ft[:, :, :self.modes1, :self.modes2, :], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2, :] += torch.einsum(
            "bixyt,ioxy->boxyt", x_ft[:, :, -self.modes1:, :self.modes2, :], self.weights2
        )

        # Inverse Fourier transform
        x = torch.fft.irfftn(out_ft, s=(size1, size2), dim=[2, 3])

        return x

def gelu(x):
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=torch.float32))))

def gelu_prime(x):
    """First derivative of GELU."""
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=torch.float32))))
    pdf = torch.exp(-0.5 * x**2) / torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.float32))
    return cdf + x * pdf

def gelu_double_prime(x):
    """Second derivative of GELU."""
    sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    exp_term = torch.exp(-0.5 * x**2)
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float32))
    return sqrt_2 * (1.0 - 0.5 * x**2) * exp_term / sqrt_pi

def gelu_triple_prime(x):
    """Third derivative of GELU."""
    sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float32))
    exp_term = torch.exp(-0.5 * x**2)
    term1 = (0.5 * sqrt_2 * x**3 * exp_term) / sqrt_pi
    term2 = (2.0 * sqrt_2 * x * exp_term) / sqrt_pi
    return term1 - term2

def gelu_quadruple_prime(x):
    """Fourth derivative of GELU."""
    sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float32))
    exp_term = torch.exp(-0.5 * x**2)
    term1 = (0.5 * sqrt_2 * x**4 * exp_term) / sqrt_pi
    term2 = (3.5 * sqrt_2 * x**2 * exp_term) / sqrt_pi
    term3 = (2.0 * sqrt_2 * exp_term) / sqrt_pi
    return -term1 + term2 - term3

class TMFNO(nn.Module):
    def __init__(self, modes1, modes2, width=62, fc_dim=128, layers=None,
                 in_dim=3, out_dim=1, activation='tanh', pad_x=0, pad_y=0, downsample_factor=2):
        """
        Fourier Neural Operator with Batched Convolutions.

        Args:
            modes1 (list): List of Fourier modes in the first dimension for each layer.
            modes2 (list): List of Fourier modes in the second dimension for each layer.
            width (int): Width of the network.
            fc_dim (int): Dimension of the fully connected layers.
            layers (list): List specifying the number of channels in each layer.
            in_dim (int): Number of input channels.
            out_dim (int): Number of output channels.
            activation (str): Activation function ('tanh', 'gelu', 'relu').
            pad_x (int): Padding in the x-direction.
            pad_y (int): Padding in the y-direction.
            downsample_factor (int): Factor by which to downsample.
        """
        super(TMFNO, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_y, 0, pad_x)
        self.num_terms = 6
        self.downsample_factor = downsample_factor

        # Downsampling and Upsampling Convolutional Layers
        self.downsample_conv = nn.Conv2d(
            layers[0], layers[0], kernel_size=3, stride=downsample_factor, padding=1, bias=False
        )
        self.upsample_conv = nn.ConvTranspose2d(
            layers[-1], layers[-1], kernel_size=3, stride=downsample_factor, padding=1,
            output_padding=downsample_factor - 1, bias=False
        )

        self.layers = layers if layers is not None else [width] * 4
        self.fc0 = nn.Linear(in_dim, layers[0])

        # Spectral convolution layers
        self.sp_convs = nn.ModuleList([
            SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)
        ])

        # Linear transformation layers
        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        ])

        # Fully connected layers
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.fcskip = nn.Linear(in_dim, layers[-1])

        # Activation function setup
        if activation == 'tanh':
            self.activation_fn = torch.tanh
            self.activation_prime_fn = lambda x: 1 - torch.tanh(x) ** 2
            self.activation_double_prime_fn = lambda x: -2 * torch.tanh(x) * (1 - torch.tanh(x) ** 2)
        elif activation == 'gelu':
            self.activation_fn = gelu
            self.activation_prime_fn = gelu_prime
            self.activation_double_prime_fn = gelu_double_prime
            self.activation_triple_prime_fn = gelu_triple_prime
            self.activation_quadruple_prime_fn = gelu_quadruple_prime
        elif activation == 'relu':
            self.activation_fn = F.relu
            self.activation_prime_fn = lambda x: (x > 0).float()
            self.activation_double_prime_fn = lambda x: torch.zeros_like(x)
        else:
            raise ValueError(f'Activation {activation} is not supported')

    def linear_transformation_broadcast(self, x, w):
        """
        Applies a 1D convolution across the last axis using an explicit loop.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width, stacked_terms).
            w (nn.Conv1d): 1D convolution layer.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        batchsize, in_channels, height, width, stacked_terms = x.shape
        out_channels = w.out_channels
        x_out = torch.zeros((batchsize, out_channels, height, width, stacked_terms), device=x.device)

        for i in range(stacked_terms):
            x_i = x[..., i].view(batchsize, in_channels, -1)
            x_out_i = w(x_i).view(batchsize, out_channels, height, width)
            x_out[..., i] = x_out_i

        return x_out

    def linear_layer_propagation(self, stacked_input, W, b):
        """
        Applies a linear transformation and propagates derivatives.

        Args:
            stacked_input (torch.Tensor): Input tensor of shape (batch, height, width, channels, total_terms).
            W (torch.Tensor): Weight matrix.
            b (torch.Tensor): Bias vector.

        Returns:
            torch.Tensor: Output tensor after linear transformation.
        """
        stacked_output = torch.einsum('bhwct,co->bhwot', stacked_input, W)

        # Add bias to the first channel
        bias_tensor = torch.zeros_like(stacked_output)
        bias_tensor[..., 0] = b
        stacked_output += bias_tensor

        return stacked_output

    def nonlinear_propagation(self, stacked_input):
        """
        Propagates derivatives through the activation function.

        Args:
            stacked_input (torch.Tensor): Input tensor of shape (batch, nx, ny, channels, total_terms).

        Returns:
            torch.Tensor: Output tensor after nonlinearity.
        """
        z = stacked_input[..., 0]
        dz_dx = stacked_input[..., 1]
        dz_dy = stacked_input[..., 2]
        d2z_dyy = stacked_input[..., 3]
        d3z_dyyy = stacked_input[..., 4]
        d4z_dyyyy = stacked_input[..., 5]

        sigma = self.activation_fn(z)
        sigma_prime = self.activation_prime_fn(z)
        sigma_double_prime = self.activation_double_prime_fn(z)
        sigma_triple_prime = self.activation_triple_prime_fn(z)
        sigma_quadruple_prime = self.activation_quadruple_prime_fn(z)

        dz_dx_old = dz_dx.clone()
        dz_dy_old = dz_dy.clone()
        d2z_dyy_old = d2z_dyy.clone()
        d3z_dyyy_old = d3z_dyyy.clone()
        d4z_dyyyy_old = d4z_dyyyy.clone()

        dz_dx = sigma_prime * dz_dx_old
        dz_dy = sigma_prime * dz_dy_old

        d2z_dyy = sigma_double_prime * dz_dy_old ** 2 + sigma_prime * d2z_dyy_old

        d3z_dyyy = (sigma_triple_prime * dz_dy_old ** 3 +
                    3 * sigma_double_prime * dz_dy_old * d2z_dyy_old +
                    sigma_prime * d3z_dyyy_old)

        d4z_dyyyy = (sigma_quadruple_prime * dz_dy_old ** 4 +
                     6 * sigma_triple_prime * dz_dy_old ** 2 * d2z_dyy_old +
                     4 * sigma_double_prime * dz_dy_old * d3z_dyyy_old +
                     3 * sigma_double_prime * d2z_dyy_old ** 2 +
                     sigma_prime * d4z_dyyyy_old)

        stacked_output = torch.stack([sigma, dz_dx, dz_dy, d2z_dyy, d3z_dyyy, d4z_dyyyy], dim=-1)

        return stacked_output

    def compute_ks_residue(self, stacked_input):
        """
        Computes the residue of the Kuramoto-Sivashinsky equation.

        Args:
            stacked_input (torch.Tensor): Input tensor containing derivatives.

        Returns:
            torch.Tensor: Residue of the KS equation.
        """
        u = stacked_input[..., 0]
        du_dx = stacked_input[..., 1]
        d2u_dx2 = stacked_input[..., 3]
        d4u_dx4 = stacked_input[..., 5]

        nonlinear_term = u * du_dx
        linear_term = d2u_dx2 + d4u_dx4

        residue = du_dx + 0.5 * nonlinear_term + linear_term

        return residue

    def forward(self, input_data):
        """
        Forward pass of the FNO model.

        Args:
            input_data (torch.Tensor): Input tensor of shape (batch, T, s, 3).

        Returns:
            torch.Tensor: Predicted output and KS residue.
        """
        batchsize, T, s, _ = input_data.shape

        # Initialize derivatives
        z = input_data.unsqueeze(-1)  # (batch, T, s, 3, 1)
        dz_dx = torch.zeros_like(z)
        dz_dy = torch.zeros_like(z)
        dz_dx[..., 2, 0] = 1.0
        dz_dy[..., 1, 0] = 1.0
        d2z_dyy = torch.zeros_like(z)
        d3z_dyyy = torch.zeros_like(z)
        d4z_dyyyy = torch.zeros_like(z)

        # Stack all derivatives
        stacked_input = torch.cat([z, dz_dx, dz_dy, d2z_dyy, d3z_dyyy, d4z_dyyyy], dim=-1)

        # Linear transformation
        x = self.linear_layer_propagation(stacked_input, self.fc0.weight.T, self.fc0.bias)
        x = x.permute(0, 3, 1, 2, 4)

        # Apply spectral convolutions and linear transformations
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = self.linear_layer_propagation(
                x.permute(0, 2, 3, 1, 4), w.weight.squeeze(2).T, w.bias
            ).permute(0, 3, 1, 2, 4)
            x = x1 + x2
            if i != len(self.sp_convs) - 1:
                x = self.nonlinear_propagation(x)

        # Final linear layers
        x = x.permute(0, 2, 3, 1, 4)
        x = self.linear_layer_propagation(x, self.fc1.weight.T, self.fc1.bias).permute(0, 3, 1, 2, 4)
        x = self.nonlinear_propagation(x).permute(0, 2, 3, 1, 4)
        x = self.linear_layer_propagation(x, self.fc2.weight.T, self.fc2.bias)

        # Extract the predicted output and compute KS residue
        return x[..., 0], self.compute_ks_residue(x)

# =========================
#         Training
# =========================

def main():
    # Argument parser for seed
    parser = argparse.ArgumentParser(description="Set random seed for reproducibility.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    seed = args.seed

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    file_path = './data/data.mat' 
    data = scipy.io.loadmat(file_path)
    U0 = torch.stack([torch.tensor(sol, dtype=torch.float32) for sol in data['init']])
    U = torch.stack([torch.tensor(sol, dtype=torch.float32) for sol in data['solutions']])

    a = U0.cpu().float()
    u = U.cpu().float()[:, :, :201, :]

    nx, nt = u.shape[-1], u.shape[1] - 1

    
    
    dataset = DataLoader1D(a, u, nx, nt)
    train_loader = dataset.make_loader(95, batch_size=1, start=0, train=True)
    test_loader = dataset.make_loader(5, batch_size=1, start=95, train=False)

    # Initialize the model
    model = TMFNO(
        modes1=[30, 30, 30, 30],
        modes2=[30, 30, 30, 30],
        fc_dim=128,
        layers=[128, 128, 128, 128, 128],
        pad_x=0,
        pad_y=0,
        activation='gelu',
    ).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    # Initialize lists to store training metrics
    train_losses = []
    validation_losses = []
    relative_l2_errors = []
    computation_times = []
    num_epochs = 150

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        model.train()

        # Training phase
        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs, residue = model(inputs)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, targets) + criterion(residue, torch.zeros_like(residue)) + 0 * criterion(outputs[:, 0, :], targets[:, 0, :])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        l2_errors = []
        epoch_val_losses = []
        with torch.no_grad():
            for input_data, final_target in test_loader:
                input_data = input_data.to(device).float()
                final_target = final_target.to(device).float()

                predicted_output, _ = model(input_data)
                predicted_output = predicted_output.squeeze(-1)

                val_loss = criterion(predicted_output, final_target).cpu().numpy()
                epoch_val_losses.append(val_loss)

                l2_error = torch.norm(predicted_output - final_target, p=2).item() / torch.norm(final_target, p=2).item()
                l2_errors.append(l2_error)

        avg_l2_error = np.mean(l2_errors)
        avg_val_loss = np.mean(epoch_val_losses)
        validation_losses.append(avg_val_loss)
        relative_l2_errors.append(avg_l2_error)

        end_time = time.time()
        computation_times.append(end_time - start_time)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, "
              f"Validation Loss: {avg_val_loss:.6f}, Relative L2 Error: {avg_l2_error:.6f}")

        # Update learning rate
        scheduler.step()

    # Save training results
    results = {
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "relative_l2_errors": relative_l2_errors,
        "computation_times": computation_times
    }

    json_filename = f"results_seed_{seed}.json"
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file)

    print(f"Training complete. Results saved to {json_filename}.")

    # Save the model
    model_filename = f"fno_model_seed_{seed}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}.")

    # Plot comparison
    def plot_comparison(predicted_output, final_target, extent=(0, 10, 0, 100)):
        """
        Plots the original and predicted solutions.

        Args:
            predicted_output (torch.Tensor): Predicted solution.
            final_target (torch.Tensor): Original solution.
            extent (tuple): Extent of the plot axes.
        """
        plt.figure(figsize=(20, 5))

        # Original solution
        plt.subplot(1, 2, 1)
        plt.imshow(final_target.cpu().numpy().T, cmap="RdBu", aspect="auto",
                   origin="lower", extent=extent, vmin=-2, vmax=2)
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Space")
        plt.title("Original Solution")

        # Predicted solution
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_output.cpu().detach().numpy().T, cmap="RdBu", aspect="auto",
                   origin="lower", extent=extent, vmin=-2, vmax=2)
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Space")
        plt.title("Model Prediction")

        plt.tight_layout()
        plt.show()

    # Visualize a sample prediction
    input_data, final_target = next(iter(test_loader))
    input_data = input_data.to(device).float()
    final_target = final_target.to(device).float()

    predicted_output, _ = model(input_data)
    predicted_output = predicted_output.squeeze(-1)

    plot_comparison(predicted_output[0], final_target[0])

if __name__ == "__main__":
    main()
