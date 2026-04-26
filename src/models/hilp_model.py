from torch import nn
from torch.types import Tensor

from .model import Model


def hilp_block(in_f: int, out_f: int) -> nn.Sequential:
    return nn.Sequential(
        HILPModel._init_layer(nn.Linear(in_f, out_f)),
        nn.GELU(),
        nn.LayerNorm(out_f)
    )


class HILPModel(Model):
    """The standard model for the HILP objective."""

    identifier = 'hilp'
    aliases = ['hilp_model', 'HILPModel', 'HILP']


    @staticmethod
    def _init_layer(layer: nn.Module) -> nn.Module:
        if hasattr(layer, 'weight') and isinstance(layer.weight, Tensor):
            nn.init.xavier_uniform_(layer.weight, gain=1)
        if hasattr(layer, 'bias') and isinstance(layer.bias, Tensor):
            nn.init.constant_(layer.bias, 0)
        return layer
    

    def _construct_model(self, input_dim: int, hidden_dims: list[int] = [512, 512], latent_dim: int = 32) -> nn.Module:
        layers = [hilp_block(input_dim, hidden_dims[0])]
        layers += [hilp_block(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        layers += [self._init_layer(nn.Linear(hidden_dims[-1], latent_dim))]
        model = nn.Sequential(*layers)
        return model
