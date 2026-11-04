from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .autograd import ReversibleSymplecticCurlFunction


class SymplecticDiffeo3D(nn.Module):
    """Volume-preserving 3D diffeomorphic layer with reversible autograd."""

    def __init__(
        self,
        steps: int = 4,
        freqs: Tuple[int, ...] = (1, 2),
        dt: float = 1.0,
        bc_mode: str = "compact_dirichlet",
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        strang_symmetric: bool = True,
        margin: float = 0.15,
        init_scale: float = 0.05
    ):
        super().__init__()
        self.steps = int(steps)
        self.freqs = tuple(int(f) for f in freqs)
        self.dt = float(dt)
        self.bc_mode = bc_mode
        self.spacing = tuple(float(s) for s in spacing)
        self.strang_symmetric = bool(strang_symmetric)
        self.margin = float(margin)

        n_freq = len(self.freqs)
        self.coeffs = nn.Parameter(
            torch.randn(self.steps, 6, n_freq, n_freq, 4) * init_scale
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        custom_coeffs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.coeffs if custom_coeffs is None else custom_coeffs
        return ReversibleSymplecticCurlFunction.apply(
            x, y, z, c,
            self.freqs, self.dt, self.bc_mode,
            self.spacing, self.strang_symmetric, self.margin
        )

    def deform_volume(
        self,
        volume: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, D, H, W = volume.shape
        device = volume.device
        dtype = volume.dtype

        if grid is None:
            gz = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
            gy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            gx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            Z, Y, X = torch.meshgrid(gz, gy, gx, indexing="ij")
        else:
            X, Y, Z = grid[..., 0], grid[..., 1], grid[..., 2]

        X_def, Y_def, Z_def = self.forward(X, Y, Z)
        deformed_grid = torch.stack([X_def, Y_def, Z_def], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)
        return F.grid_sample(volume, deformed_grid, mode="bilinear", padding_mode="border", align_corners=True)


def create_normalized_grid_3d(
    shape: Tuple[int, int, int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D, H, W = shape
    gz = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
    gy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    gx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    Z, Y, X = torch.meshgrid(gz, gy, gx, indexing="ij")
    return X, Y, Z
