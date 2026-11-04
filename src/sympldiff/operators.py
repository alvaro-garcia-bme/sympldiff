"""
Symplectic Coordinate Shear Operators & Mathematical Stream Bases
=================================================================
Implements the Lie-Helmholtz volume-preserving coordinate shear splitting on
3D domains Omega = [-1, 1]^3.

Each shear stage S_k: R^3 -> R^3 updates a single active coordinate using a smooth
potential stream function of the two passive coordinates:
    S_1: (x, y, z) -> (x,  y + dt * f_1(x, z),  z)
    S_2: (x, y, z) -> (x,  y,  z + dt * f_2(x, y))
    S_3: (x, y, z) -> (x,  y,  z + dt * f_3(y, x))
    S_4: (x, y, z) -> (x + dt * f_4(y, z),  y,  z)
    S_5: (x, y, z) -> (x + dt * f_5(z, y),  y,  z)
    S_6: (x, y, z) -> (x,  y + dt * f_6(z, x),  z)

Every stage has a triangular Jacobian matrix with unit diagonal elements, yielding:
    det(D S_k(x)) === 1.000000000000000
to exact IEEE-754 machine precision without numerical Taylor approximations.
"""

from typing import Tuple, List, Union, Optional
import numpy as np
import torch
from .boundary import boundary_envelope_np, boundary_envelope_torch


def eval_sobolev_stream_2d_np(
    u: np.ndarray,
    v: np.ndarray,
    coeffs: np.ndarray,
    freqs: Tuple[int, ...] = (1, 2),
    bc_mode: str = "compact_dirichlet",
    aniso_scale: float = 1.0,
    margin: float = 0.15,
    need_grad: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Evaluates 2D Sobolev H^2-normalized trigonometric stream function f(u, v)
    and its analytical derivatives in NumPy.

    Parameters
    ----------
    u, v : np.ndarray
        Passive coordinate arrays.
    coeffs : np.ndarray
        Fourier coefficient array of shape (len(freqs), len(freqs), 4).
    freqs : tuple of int
        Harmonic frequencies (default (1, 2)).
    bc_mode : str
        Boundary mode ('periodic', 'dirichlet', or 'compact_dirichlet').
    aniso_scale : float
        Anisotropic metric scaling factor.
    margin : float
        Safety margin for compact Dirichlet boundary.
    need_grad : bool
        Whether to compute analytical partial derivatives df/du and df/dv.

    Returns
    -------
    f : np.ndarray
        Evaluated stream potential.
    df_du, df_dv : np.ndarray or None
        Analytical derivatives.
    """
    H = np.zeros_like(u, dtype=np.float64)
    dH_du = np.zeros_like(u, dtype=np.float64) if need_grad else None
    dH_dv = np.zeros_like(u, dtype=np.float64) if need_grad else None
    pi = np.pi
    sv_list = [np.sin(fy * pi * v) for fy in freqs]
    cv_list = [np.cos(fy * pi * v) for fy in freqs]

    for i, fx in enumerate(freqs):
        ku = fx * pi
        su = np.sin(ku * u)
        cu = np.cos(ku * u)
        for j, fy in enumerate(freqs):
            kv = fy * pi
            scale = float(aniso_scale) / (pi * np.sqrt(float(fx ** 2 + fy ** 2)))
            sv = sv_list[j]
            cv = cv_list[j]
            c0, c1, c2, c3 = coeffs[i, j] * scale
            H += c0 * su * sv + c1 * cu * cv + c2 * su * cv + c3 * cu * sv
            if need_grad:
                dH_du += ku * (c0 * cu * sv - c1 * su * cv + c2 * cu * cv - c3 * su * sv)
                dH_dv += kv * (c0 * su * cv - c1 * cu * sv - c2 * su * sv + c3 * cu * cv)

    if bc_mode in ("dirichlet", "compact_dirichlet"):
        if not need_grad:
            w, _, _ = boundary_envelope_np(u, v, bc_mode=bc_mode, margin=margin)
            return w * H, None, None
        w, dw_du, dw_dv = boundary_envelope_np(u, v, bc_mode=bc_mode, margin=margin)
        dH_du_env = dw_du * H + w * dH_du
        dH_dv_env = dw_dv * H + w * dH_dv
        return w * H, dH_du_env, dH_dv_env

    return H, dH_du, dH_dv


def eval_sobolev_stream_2d_torch(
    u: torch.Tensor,
    v: torch.Tensor,
    coeffs: torch.Tensor,
    freqs: Tuple[int, ...] = (1, 2),
    bc_mode: str = "compact_dirichlet",
    aniso_scale: float = 1.0,
    margin: float = 0.15
) -> torch.Tensor:
    """
    Evaluates 2D Sobolev H^2-normalized trigonometric stream function in PyTorch.
    Supports autograd through both coordinates and coefficients.
    """
    pi = torch.tensor(np.pi, device=u.device, dtype=u.dtype)
    H = torch.zeros_like(u)
    sv_list = [torch.sin(fy * pi * v) for fy in freqs]
    cv_list = [torch.cos(fy * pi * v) for fy in freqs]

    for i, fx in enumerate(freqs):
        ku = fx * pi
        su = torch.sin(ku * u)
        cu = torch.cos(ku * u)
        for j, fy in enumerate(freqs):
            kv = fy * pi
            scale = float(aniso_scale) / (np.pi * np.sqrt(float(fx ** 2 + fy ** 2)))
            sv = sv_list[j]
            cv = cv_list[j]
            c = coeffs[i, j] * scale
            H = H + c[0] * su * sv + c[1] * cu * cv + c[2] * su * cv + c[3] * cu * sv

    if bc_mode in ("dirichlet", "compact_dirichlet"):
        w = boundary_envelope_torch(u, v, bc_mode=bc_mode, margin=margin)
        return w * H

    return H


class SymplecticCurlDiffeo3D_NP:
    """
    NumPy reference implementation of 3D Symplectic Diffeomorphic Splitting.
    Provides forward mapping, exact closed-form inverse, and exact analytical Jacobian.
    """

    def __init__(
        self,
        freqs: Tuple[int, ...] = (1, 2),
        dt: float = 1.0,
        bc_mode: str = "compact_dirichlet",
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        strang_symmetric: bool = True,
        margin: float = 0.15
    ):
        self.freqs = freqs
        self.dt = float(dt)
        self.bc_mode = bc_mode
        self.spacing = tuple(float(s) for s in spacing)
        self.strang_symmetric = bool(strang_symmetric)
        self.margin = float(margin)

        sx, sy, sz = self.spacing
        s_mean = (sx + sy + sz) / 3.0
        self.aniso_scales = (
            s_mean / sy,
            s_mean / sz,
            s_mean / sz,
            s_mean / sx,
            s_mean / sx,
            s_mean / sy,
        )

    def _single_stage_forward(self, x, y, z, stage_idx, c_stage, dt_stage):
        sc = self.aniso_scales[stage_idx]
        if stage_idx == 0:
            f, _, _ = eval_sobolev_stream_2d_np(x, z, c_stage, self.freqs, self.bc_mode, sc, self.margin, need_grad=False)
            y = y + dt_stage * f
        elif stage_idx == 1:
            f, _, _ = eval_sobolev_stream_2d_np(x, y, c_stage, self.freqs, self.bc_mode, sc, self.margin, need_grad=False)
            z = z + dt_stage * f
        elif stage_idx == 2:
            f, _, _ = eval_sobolev_stream_2d_np(y, x, c_stage, self.freqs, self.bc_mode, sc, self.margin, need_grad=False)
            z = z + dt_stage * f
        elif stage_idx == 3:
            f, _, _ = eval_sobolev_stream_2d_np(y, z, c_stage, self.freqs, self.bc_mode, sc, self.margin, need_grad=False)
            x = x + dt_stage * f
        elif stage_idx == 4:
            f, _, _ = eval_sobolev_stream_2d_np(z, y, c_stage, self.freqs, self.bc_mode, sc, self.margin, need_grad=False)
            x = x + dt_stage * f
        elif stage_idx == 5:
            f, _, _ = eval_sobolev_stream_2d_np(z, x, c_stage, self.freqs, self.bc_mode, sc, self.margin, need_grad=False)
            y = y + dt_stage * f
        return x, y, z

    def forward(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, coeffs_list: Union[np.ndarray, List[np.ndarray]]):
        """
        Computes forward diffeomorphic deformation: (X, Y, Z) -> (x, y, z).
        """
        if isinstance(coeffs_list, np.ndarray) and coeffs_list.ndim == 4:
            coeffs_list = [coeffs_list]
        x, y, z = X.astype(np.float64).copy(), Y.astype(np.float64).copy(), Z.astype(np.float64).copy()

        for c_step in coeffs_list:
            if self.strang_symmetric:
                half_dt = 0.5 * self.dt
                for s_idx in range(6):
                    x, y, z = self._single_stage_forward(x, y, z, s_idx, c_step[s_idx], half_dt)
                for s_idx in reversed(range(6)):
                    x, y, z = self._single_stage_forward(x, y, z, s_idx, c_step[s_idx], half_dt)
            else:
                for s_idx in range(6):
                    x, y, z = self._single_stage_forward(x, y, z, s_idx, c_step[s_idx], self.dt)
        return x, y, z

    def inverse(self, X_new: np.ndarray, Y_new: np.ndarray, Z_new: np.ndarray, coeffs_list: Union[np.ndarray, List[np.ndarray]]):
        """
        Computes exact O(1) closed-form analytical inverse: (X_new, Y_new, Z_new) -> (X, Y, Z).
        Requires zero numerical inversion or grid interpolation.
        """
        if isinstance(coeffs_list, np.ndarray) and coeffs_list.ndim == 4:
            coeffs_list = [coeffs_list]
        x, y, z = X_new.astype(np.float64).copy(), Y_new.astype(np.float64).copy(), Z_new.astype(np.float64).copy()

        for c_step in reversed(coeffs_list):
            if self.strang_symmetric:
                half_dt = -0.5 * self.dt
                for s_idx in range(6):
                    x, y, z = self._single_stage_forward(x, y, z, s_idx, c_step[s_idx], half_dt)
                for s_idx in reversed(range(6)):
                    x, y, z = self._single_stage_forward(x, y, z, s_idx, c_step[s_idx], half_dt)
            else:
                for s_idx in reversed(range(6)):
                    x, y, z = self._single_stage_forward(x, y, z, s_idx, c_step[s_idx], -self.dt)
        return x, y, z

    def exact_jacobian(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, coeffs_list: Union[np.ndarray, List[np.ndarray]]):
        """
        Evaluates the exact analytical Jacobian matrix (N, 3, 3) along the deformation trajectory.
        """
        if isinstance(coeffs_list, np.ndarray) and coeffs_list.ndim == 4:
            coeffs_list = [coeffs_list]
        N = X.size
        x, y, z = X.ravel().astype(np.float64).copy(), Y.ravel().astype(np.float64).copy(), Z.ravel().astype(np.float64).copy()
        J = np.zeros((N, 3, 3), dtype=np.float64)
        J[:, 0, 0] = 1.0
        J[:, 1, 1] = 1.0
        J[:, 2, 2] = 1.0

        def apply_stage_jac(x, y, z, J, s_idx, c_stage, dt_s):
            sc = self.aniso_scales[s_idx]
            if s_idx == 0:
                f, df_dx, df_dz = eval_sobolev_stream_2d_np(x, z, c_stage, self.freqs, self.bc_mode, sc, self.margin)
                y = y + dt_s * f
                J[:, 1, :] += dt_s * (df_dx[:, None] * J[:, 0, :] + df_dz[:, None] * J[:, 2, :])
            elif s_idx == 1:
                f, df_dx, df_dy = eval_sobolev_stream_2d_np(x, y, c_stage, self.freqs, self.bc_mode, sc, self.margin)
                z = z + dt_s * f
                J[:, 2, :] += dt_s * (df_dx[:, None] * J[:, 0, :] + df_dy[:, None] * J[:, 1, :])
            elif s_idx == 2:
                f, df_dy, df_dx = eval_sobolev_stream_2d_np(y, x, c_stage, self.freqs, self.bc_mode, sc, self.margin)
                z = z + dt_s * f
                J[:, 2, :] += dt_s * (df_dy[:, None] * J[:, 1, :] + df_dx[:, None] * J[:, 0, :])
            elif s_idx == 3:
                f, df_dy, df_dz = eval_sobolev_stream_2d_np(y, z, c_stage, self.freqs, self.bc_mode, sc, self.margin)
                x = x + dt_s * f
                J[:, 0, :] += dt_s * (df_dy[:, None] * J[:, 1, :] + df_dz[:, None] * J[:, 2, :])
            elif s_idx == 4:
                f, df_dz, df_dy = eval_sobolev_stream_2d_np(z, y, c_stage, self.freqs, self.bc_mode, sc, self.margin)
                x = x + dt_s * f
                J[:, 0, :] += dt_s * (df_dz[:, None] * J[:, 2, :] + df_dy[:, None] * J[:, 1, :])
            elif s_idx == 5:
                f, df_dz, df_dx = eval_sobolev_stream_2d_np(z, x, c_stage, self.freqs, self.bc_mode, sc, self.margin)
                y = y + dt_s * f
                J[:, 1, :] += dt_s * (df_dz[:, None] * J[:, 2, :] + df_dx[:, None] * J[:, 0, :])
            return x, y, z, J

        for c_step in coeffs_list:
            if self.strang_symmetric:
                half_dt = 0.5 * self.dt
                for s_idx in range(6):
                    x, y, z, J = apply_stage_jac(x, y, z, J, s_idx, c_step[s_idx], half_dt)
                for s_idx in reversed(range(6)):
                    x, y, z, J = apply_stage_jac(x, y, z, J, s_idx, c_step[s_idx], half_dt)
            else:
                for s_idx in range(6):
                    x, y, z, J = apply_stage_jac(x, y, z, J, s_idx, c_step[s_idx], self.dt)

        return J.reshape(X.shape + (3, 3))
