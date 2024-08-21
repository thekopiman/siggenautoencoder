import torch
import torch.nn as nn
import torch.optim as optim


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, seq_length):
        super(LSTMAutoencoder, self).__init__()
        self.BN_encoder = nn.BatchNorm1d(seq_length)
        self.encoder = nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True)
        self.BN_latent = nn.BatchNorm1d(latent_dim)
        
        self.decoder = nn.LSTM(latent_dim, input_dim, num_layers, batch_first=True)
        self.BN_decoder = nn.BatchNorm1d(seq_length)

        self.latent_dim = latent_dim
        self.seq_length = seq_length

    def forward(self, x):
        # Encode
        x = self.BN_encoder(x)
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]  # Get the hidden state from the last layer
        after_bn = self.BN_latent(latent)

        # Prepare repeated latent vector for decoder input
        latent_repeated = after_bn.unsqueeze(1).repeat(1, self.seq_length, 1)

        # Decode
        output, _ = self.decoder(latent_repeated)
        output = self.BN_decoder(output)

        return output, after_bn

