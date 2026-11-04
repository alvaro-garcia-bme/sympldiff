"""
High-Level PyTorch Neural Network Layers and Spatial Transformers
================================================================
Provides `SymplecticDiffeo3D`, an `nn.Module` that can be dropped directly into
any deep learning registration pipeline (e.g., VoxelMorph, UNet, or Spatial Transformers).
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .autograd import ReversibleSymplecticCurlFunction


class SymplecticDiffeo3D(nn.Module):
    """
    Volume-Preserving Diffeomorphic Transformation Layer with O(1) Reversible Autograd.

    Parameters
    ----------
    steps : int
        Number of composition steps L (default: 4).
    freqs : tuple of int
        Fourier harmonic frequencies for stream functions (default: (1, 2)).
    dt : float
        Integration time step (default: 1.0).
    bc_mode : str
        Boundary mode: 'compact_dirichlet', 'dirichlet', or 'periodic'.
    spacing : tuple of float
        Physical voxel spacings (dx, dy, dz).
    strang_symmetric : bool
        If True, applies 12-stage palindromic Strang composition (O(dt^2) convergence).
    margin : float
        Compact boundary margin in [-1, 1] domain.
    """

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
        # Learnable shear stream parameters: (steps, 6 stages, n_freq, n_freq, 4 harmonic modes)
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
        """
        Applies forward diffeomorphic mapping: (x, y, z) -> (x_def, y_def, z_def).
        """
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
        """
        Deforms a 3D image volume of shape (B, C, D, H, W) through the volume-preserving diffeomorphism.

        Parameters
        ----------
        volume : torch.Tensor
            Input 3D volume tensor (B, C, D, H, W) on [-1, 1]^3.
        grid : torch.Tensor, optional
            Pre-computed identity coordinate grid of shape (D, H, W, 3). If None, generated automatically.

        Returns
        -------
        deformed : torch.Tensor
            Warped 3D volume tensor (B, C, D, H, W).
        """
        B, C, D, H, W = volume.shape
        device = volume.device
        dtype = volume.dtype

        if grid is None:
            # Create canonical normalized [-1, 1] grid
            gz = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
            gy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            gx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            Z, Y, X = torch.meshgrid(gz, gy, gx, indexing="ij")
        else:
            X, Y, Z = grid[..., 0], grid[..., 1], grid[..., 2]

        X_def, Y_def, Z_def = self.forward(X, Y, Z)

        # PyTorch grid_sample expects (X, Y, Z) order in the last dimension
        deformed_grid = torch.stack([X_def, Y_def, Z_def], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)
        return F.grid_sample(volume, deformed_grid, mode="bilinear", padding_mode="border", align_corners=True)


def create_normalized_grid_3d(
    shape: Tuple[int, int, int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates a 3D coordinate meshgrid on [-1, 1]^3.
    """
    D, H, W = shape
    gz = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
    gy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    gx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    Z, Y, X = torch.meshgrid(gz, gy, gx, indexing="ij")
    return X, Y, Z
