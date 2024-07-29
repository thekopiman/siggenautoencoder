import torch
import torch.nn.functional as F
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTMAutoencoder

    Ensure that the model is always batched.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_size,  # 2 for IQ
        hidden_size,  # Can test out 2 to 4 - LSTM blocks per layer
        num_layers,  # Layer stacks
        bias: bool = True,
        batch_first: bool = True,
        dropout: int = 0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ):
        super().__init__()

        self.batch_first = batch_first

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

    def forward(self, x):
        if self.batch_first:
            seq_length = x.shape[1]
        else:
            seq_length = x.shape[0]

        _, (hidden, _) = self.encoder(x)

        latent = hidden[-1]  # Obtain the last hidden layer only

        # Prepare repeated latent vector for decoder input
        latent_repeated = latent.unsqueeze(1).repeat(1, seq_length, 1)

        # Decode
        output, _ = self.decoder(latent_repeated)

        return output, latent
