import torch
import torch.nn.functional as F
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """DenseAutoencoder

    Ensure that the model is always batched.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_size: int = 5000, 
        latent_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5000, 3000),
            nn.Linear(3000, 200),
            nn.Linear(200, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,200),
            nn.Linear(200, 3000),
            nn.Linear(3000,5000)
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        
        output = output.unsqueeze(-1)
        
        return output, latent
