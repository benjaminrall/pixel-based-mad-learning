from torch import nn
from torch.types import Tensor

from .model import Model


def mad_block(in_f: int, out_f: int) -> nn.Sequential:
    return nn.Sequential(
        AtariMadDistModel._init_layer(nn.Linear(in_f, out_f)),
        nn.SELU(),
    )


class AtariMadDistModel(Model):
    """The model used to train the MadDist objective on pixel-based Atari 2600 observations."""

    identifier = 'atari_maddist'
    aliases = ['atari_maddist_model', 'AtariMadDistModel', 'AtariMadDist', 'atari_mad']


    @staticmethod
    def _init_layer(layer: nn.Module) -> nn.Module:
        if hasattr(layer, 'weight') and isinstance(layer.weight, Tensor):
            nn.init.xavier_uniform_(layer.weight, gain=1)
        if hasattr(layer, 'bias') and isinstance(layer.bias, Tensor):
            nn.init.constant_(layer.bias, 0)
        return layer


    def _construct_model(self, _: int, hidden_dims: list[int] = [512, 512, 256], latent_dim: int = 512) -> nn.Module:
        layers = [
            self._init_layer(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        ]
        layers += [mad_block(64 * 7 * 7, hidden_dims[0])]
        layers += [mad_block(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        layers += [self._init_layer(nn.Linear(hidden_dims[-1], latent_dim))]
        model = nn.Sequential(*layers)
        return model
    

    def forward(self, x: Tensor) -> Tensor:
        # Maps inputs to range [0, 1]
        x = x / 255.0        
        return self.model(x)    
