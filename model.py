import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder3d(nn.Module):
    def __init__(self, rows, cols, in_channels):
        super(Encoder3d, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(rows * cols * in_channels * 20, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 200)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(256, 200)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # x = self.relu3(x)
        # x = self.fc4(x)
        return x
        
class Decoder2d(nn.Module):
    def __init__(self, in_dims, in_channels, hidden_channels, out_channels):
        super(Decoder2d, self).__init__()
        self.in_channels = in_channels
        self.head = nn.Linear(in_dims, in_channels*(32**2))
        self.convt1 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2)
        self.convt2 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.head(x)
        x = x.reshape(x.shape[0],self.in_channels, 32, 32)
        out = F.relu(self.convt1(x))
        out = torch.sigmoid(self.convt2(out))
        return out

class RLNetwork(nn.Module):
    def __init__(self, rows, cols, in_channels, out_channels_enc, hidden_channels_dec) -> None:
        super(RLNetwork, self).__init__()
        self.encoder = Encoder3d(rows, cols, in_channels)
        
        self.head_mu = nn.Linear(200, 200)
        self.head_logvar = nn.Linear(200, 200)

        self.decoder_e = Decoder2d(200, out_channels_enc, hidden_channels_dec, in_channels)
        self.decoder_a = Decoder2d(200, out_channels_enc, hidden_channels_dec, in_channels)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        return mu

    def forward(self, data):
        x = self.encoder(data)
        # x_flat = x.flatten(1)
        mu = self.head_mu(x)
        logvar = self.head_logvar(x)
        z = self.reparameterize(mu, logvar)

        recon_grid = self.decoder_e(z)
        recon_agent = self.decoder_a(z)
        return recon_grid, recon_agent, mu, logvar, z