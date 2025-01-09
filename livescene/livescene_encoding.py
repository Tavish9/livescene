from typing import Literal, Sequence

import torch
from jaxtyping import Float
from nerfstudio.field_components import Encoding
from torch import Tensor, nn


# Adapted from nerfstudio.field_components.encodings.KPlanesEncoding
class LiveSceneEncoding(Encoding):
    def __init__(
        self,
        resolution: Sequence[int] = (128, 128, 128),
        num_components: int = 64,
        init_a: float = 0,
        init_b: float = 1,
        reduce: Literal["sum", "product", "none"] = "none",
    ) -> None:
        super().__init__(in_dim=len(resolution))

        self.resolution = resolution
        self.num_components = num_components
        self.reduce = reduce
        self.plane_coefs = nn.ParameterList()
        for res in self.resolution:
            grid = nn.Parameter(torch.empty(1, res + 1, self.num_components))
            nn.init.uniform_(grid, a=init_a, b=init_b)
            self.plane_coefs.append(grid)

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs 1"]) -> Float[Tensor, "*bs output_dim"]:
        """Sample features from this encoder. Expects ``in_tensor`` to be in range [0, 1]"""
        output = torch.zeros(0, in_tensor.shape[0], self.num_components, device=in_tensor.device)
        for i, grid in enumerate(self.plane_coefs):
            coords = self.resolution[i] * in_tensor.reshape(-1)
            codf, codc = coords.floor().long(), coords.ceil().long()
            factor = (coords - codf).reshape(1, -1, 1)
            feature = (1 - factor) * grid[:, codf] + factor * grid[:, codc]
            output = torch.cat([output, feature], dim=0)
        if self.reduce == "product":
            output = output.prod(dim=0, keepdim=True)
        elif self.reduce == "sum":
            output = output.sum(dim=0, keepdim=True)
        return output.permute(1, 0, 2)
