import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    """LSTMAutoencoder

    Ensure that the model is always batched.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_size: int = 2,  # 2 for IQ
        latent_dim: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_size, 16, kernel_size=3, stride=2, padding=1
            ),  # (batch_size, 16, sequence_length/2)
            nn.ReLU(),
            nn.Conv1d(
                16, 32, kernel_size=3, stride=2, padding=1
            ),  # (batch_size, 32, sequence_length/4)
            nn.ReLU(),
            nn.Conv1d(
                32, latent_dim, kernel_size=3, stride=2, padding=1
            ),  # (batch_size, latent_dim, sequence_length/8)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch_size, 32, sequence_length/4)
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch_size, 16, sequence_length/2)
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, input_size, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch_size, input_size, sequence_length)
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output, latent
