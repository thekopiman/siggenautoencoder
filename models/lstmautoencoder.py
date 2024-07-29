import torch
import torch.nn as nn
import torch.optim as optim


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, seq_length):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, input_dim, num_layers, batch_first=True)
        self.latent_dim = latent_dim
        self.seq_length = seq_length

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]  # Get the hidden state from the last layer

        # Prepare repeated latent vector for decoder input
        latent_repeated = latent.unsqueeze(1).repeat(1, self.seq_length, 1)

        # Decode
        output, _ = self.decoder(latent_repeated)

        return output, latent


# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     output, latent = model(x_train)
#     loss = criterion(output, x_train)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
