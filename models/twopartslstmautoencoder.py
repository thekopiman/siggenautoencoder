import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, seq_length):
            super(Encoder, self).__init__()
            self.encoder = nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True)
            self.BN_latent = nn.BatchNorm1d(latent_dim)

            self.latent_dim = latent_dim

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]  # Get the hidden state from the last layer
        after_bn = self.BN_latent(latent)
        
        return after_bn

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, seq_length):
        super(Decoder, self).__init__()
        self.decoder = nn.LSTM(latent_dim, input_dim, num_layers, batch_first=True)
        
        self.seq_length = seq_length
        self.latent_dim = latent_dim

    def forward(self, x):
        # Prepare repeated latent vector for decoder input
        latent_repeated = x.unsqueeze(1).repeat(1, self.seq_length, 1)

        # Decode
        output, _ = self.decoder(latent_repeated)

        return output
        
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, seq_length):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, num_layers, seq_length)
        self.decoder = Decoder(input_dim, latent_dim, num_layers, seq_length)

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        y = self.decoder(x)

        return y, x

