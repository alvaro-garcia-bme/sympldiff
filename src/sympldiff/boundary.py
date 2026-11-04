from typing import Tuple
import numpy as np
import torch


def boundary_envelope_np(
    u: np.ndarray,
    v: np.ndarray,
    bc_mode: str = "compact_dirichlet",
    margin: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C^2 smooth boundary envelope on the passive coordinates."""
    if bc_mode == "dirichlet":
        uc = np.clip(u, -1.0, 1.0)
        vc = np.clip(v, -1.0, 1.0)
        wu = np.cos(0.5 * np.pi * uc) ** 2
        wv = np.cos(0.5 * np.pi * vc) ** 2
        du_wu = -0.5 * np.pi * np.sin(np.pi * uc) * (np.abs(u) <= 1.0)
        dv_wv = -0.5 * np.pi * np.sin(np.pi * vc) * (np.abs(v) <= 1.0)
        return wu * wv, du_wu * wv, wu * dv_wv

    elif bc_mode == "compact_dirichlet":
        L = 1.0 - float(margin)
        us = np.clip(u / L, -1.0, 1.0)
        vs = np.clip(v / L, -1.0, 1.0)
        wu = np.cos(0.5 * np.pi * us) ** 2
        wv = np.cos(0.5 * np.pi * vs) ** 2
        du_wu = (-0.5 * np.pi / L) * np.sin(np.pi * us) * (np.abs(u) <= L)
        dv_wv = (-0.5 * np.pi / L) * np.sin(np.pi * vs) * (np.abs(v) <= L)
        return wu * wv, du_wu * wv, wu * dv_wv

    ones = np.ones_like(u)
    zeros = np.zeros_like(u)
    return ones, zeros, zeros


def boundary_envelope_torch(
    u: torch.Tensor,
    v: torch.Tensor,
    bc_mode: str = "compact_dirichlet",
    margin: float = 0.15
) -> torch.Tensor:
    """C^2 smooth boundary envelope on the passive coordinates in PyTorch."""
    if bc_mode == "dirichlet":
        uc = torch.clamp(u, -1.0, 1.0)
        vc = torch.clamp(v, -1.0, 1.0)
        return (torch.cos(0.5 * np.pi * uc) ** 2) * (torch.cos(0.5 * np.pi * vc) ** 2)

    elif bc_mode == "compact_dirichlet":
        L = 1.0 - float(margin)
        us = torch.clamp(u / L, -1.0, 1.0)
        vs = torch.clamp(v / L, -1.0, 1.0)
        return (torch.cos(0.5 * np.pi * us) ** 2) * (torch.cos(0.5 * np.pi * vs) ** 2)

    return torch.ones_like(u)
