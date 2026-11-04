"""
Boundary Envelope and Confinement Functions for Sympldiff
=========================================================
Implements smooth C^2 boundary envelopes w(u, v) depending only on the
passive sheared coordinate directions. Because w(u, v) is strictly independent
of the active coordinate being sheared, multiplying the stream function f(u, v)
by w(u, v) leaves the active partial derivative identically zero:
    d(x_active)/d(x_active) === 1.000000000000000
thus guaranteeing exact unit Jacobian determinant and exact O(1) closed-form inversion.
"""

from typing import Tuple
import numpy as np
import torch


def boundary_envelope_np(
    u: np.ndarray,
    v: np.ndarray,
    bc_mode: str = "compact_dirichlet",
    margin: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates 2D passive-coordinate boundary envelope and analytical gradients in NumPy.

    Parameters
    ----------
    u, v : np.ndarray
        Passive planar coordinate grids on [-1, 1].
    bc_mode : str
        Boundary mode: 'periodic', 'dirichlet', or 'compact_dirichlet'.
    margin : float
        Safety boundary margin for 'compact_dirichlet' (domain [-1+margin, 1-margin]).

    Returns
    -------
    w : np.ndarray
        Boundary envelope weight.
    dw_du : np.ndarray
        Analytical derivative with respect to u.
    dw_dv : np.ndarray
        Analytical derivative with respect to v.
    """
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

    else:  # periodic
        ones = np.ones_like(u)
        zeros = np.zeros_like(u)
        return ones, zeros, zeros


def boundary_envelope_torch(
    u: torch.Tensor,
    v: torch.Tensor,
    bc_mode: str = "compact_dirichlet",
    margin: float = 0.15
) -> torch.Tensor:
    """
    Evaluates 2D passive-coordinate boundary envelope in PyTorch.

    Parameters
    ----------
    u, v : torch.Tensor
        Passive coordinate tensors on [-1, 1].
    bc_mode : str
        Boundary mode: 'periodic', 'dirichlet', or 'compact_dirichlet'.
    margin : float
        Safety boundary margin for 'compact_dirichlet'.

    Returns
    -------
    w : torch.Tensor
        Boundary envelope weight tensor.
    """
    if bc_mode == "dirichlet":
        uc = torch.clamp(u, -1.0, 1.0)
        vc = torch.clamp(v, -1.0, 1.0)
        return (torch.cos(0.5 * np.pi * uc) ** 2) * (torch.cos(0.5 * np.pi * vc) ** 2)

    elif bc_mode == "compact_dirichlet":
        L = 1.0 - float(margin)
        us = torch.clamp(u / L, -1.0, 1.0)
        vs = torch.clamp(v / L, -1.0, 1.0)
        return (torch.cos(0.5 * np.pi * us) ** 2) * (torch.cos(0.5 * np.pi * vs) ** 2)

    else:  # periodic
        return torch.ones_like(u)
