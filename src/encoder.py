import torch.nn as nn

from .utils import orthogonal_init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class Impala(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), nn.ReLU(), nn.BatchNorm2d(32),
            Flatten(),
            nn.Linear(in_features=288, out_features=feature_dim), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=1, bidirectional=False)
        self.apply(orthogonal_init)

    def forward(self, x, hidden_states=None):
        x = self.layers(x)
        x = x.view(x.size(0), 1, -1)
        if hidden_states:
            x, hidden_states = self.lstm(x, hidden_states)
        else:
            x, hidden_states = self.lstm(x)
        x = x.view(x.size(0), -1)
        return x, hidden_states
