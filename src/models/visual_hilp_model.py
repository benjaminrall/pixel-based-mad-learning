from torch import nn
from torch.types import Tensor

from .model import Model


def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        VisualHILPModel._init_layer(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)),
        nn.ReLU()
    )

def hilp_block(in_f: int, out_f: int) -> nn.Sequential:
    return nn.Sequential(
        VisualHILPModel._init_layer(nn.Linear(in_f, out_f)),
        nn.GELU(),
        nn.LayerNorm(out_f)
    )


class VisualHILPModel(Model):
    """The model used to train the HILP objective on pixel-based AntMaze observations."""

    identifier = 'visual_hilp'
    aliases = ['visual_hilp_model', 'VisualHILPModel', 'VisualHILP']


    @staticmethod
    def _init_layer(layer: nn.Module) -> nn.Module:
        if hasattr(layer, 'weight') and isinstance(layer.weight, Tensor):
            nn.init.xavier_uniform_(layer.weight, gain=1)
        if hasattr(layer, 'bias') and isinstance(layer.bias, Tensor):
            nn.init.constant_(layer.bias, 0)
        return layer
    

    def _construct_model(self, input_dim: int, hidden_dims: list[int] = [512, 512], latent_dim: int = 32) -> nn.Module:
        layers = [
            conv_block(input_dim, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.Flatten(),
        ]
        layers += [hilp_block(256*4*4, hidden_dims[0])]
        layers += [hilp_block(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        layers += [self._init_layer(nn.Linear(hidden_dims[-1], latent_dim))]
        model = nn.Sequential(*layers)
        return model
    

    def forward(self, x: Tensor) -> Tensor:
        # Maps inputs to range [0, 1] and ensures correct channel ordering
        if x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        x = x / 255.0        
        return self.model(x)
