"""
Evaluation Metrics, Numerical Verification, and Geometric Invariants
====================================================================
Provides evaluation functions for:
1. Exact Analytical and Finite-Difference Jacobian Determinants.
2. Canonical Kuhn-Freudenthal 6-Tetrahedral Sub-Voxel Determinants.
3. Topological Folding Percentages (% of det(J) <= 0).
4. Jacobian Condition Numbers kappa_2(D Phi).
5. Sobolev H^2 Regularization Norms.
"""

from typing import Tuple
import numpy as np


def compute_finite_difference_jacobian_det_3d(
    X_def: np.ndarray,
    Y_def: np.ndarray,
    Z_def: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    indexing: str = "ij"
) -> np.ndarray:
    """
    Computes numerical Jacobian determinant via 2nd-order central finite differences.

    Parameters
    ----------
    X_def, Y_def, Z_def : np.ndarray
        Deformed coordinate 3D arrays of shape (N1, N2, N3).
    voxel_spacing : tuple of float
        Spacing (dx, dy, dz).
    indexing : str
        'ij' (axis 0=x, 1=y, 2=z) or 'xy' (axis 0=y, 1=x, 2=z).

    Returns
    -------
    det_J : np.ndarray
        Jacobian determinant array.
    """
    dx, dy, dz = voxel_spacing
    if indexing == "ij":
        dPhi_x_dx, dPhi_x_dy, dPhi_x_dz = np.gradient(X_def, dx, dy, dz)
        dPhi_y_dx, dPhi_y_dy, dPhi_y_dz = np.gradient(Y_def, dx, dy, dz)
        dPhi_z_dx, dPhi_z_dy, dPhi_z_dz = np.gradient(Z_def, dx, dy, dz)
    else:
        dPhi_x_dy, dPhi_x_dx, dPhi_x_dz = np.gradient(X_def, dy, dx, dz)
        dPhi_y_dy, dPhi_y_dx, dPhi_y_dz = np.gradient(Y_def, dy, dx, dz)
        dPhi_z_dy, dPhi_z_dx, dPhi_z_dz = np.gradient(Z_def, dy, dx, dz)

    # Determinant of 3x3 matrix:
    # [ dPhi_x_dx, dPhi_x_dy, dPhi_x_dz ]
    # [ dPhi_y_dx, dPhi_y_dy, dPhi_y_dz ]
    # [ dPhi_z_dx, dPhi_z_dy, dPhi_z_dz ]
    det_J = (
        dPhi_x_dx * (dPhi_y_dy * dPhi_z_dz - dPhi_y_dz * dPhi_z_dy)
        - dPhi_x_dy * (dPhi_y_dx * dPhi_z_dz - dPhi_y_dz * dPhi_z_dx)
        + dPhi_x_dz * (dPhi_y_dx * dPhi_z_dy - dPhi_y_dy * dPhi_z_dx)
    )
    return det_J


def compute_folding_percentage(det_J: np.ndarray) -> float:
    """
    Computes percentage of volume elements undergoing topological singularity (det(J) <= 0).
    A valid diffeomorphism has strictly 0.0000% folding.
    """
    num_folded = np.sum(det_J <= 0.0)
    return float(100.0 * num_folded / det_J.size)


def compute_jacobian_condition_number(J: np.ndarray) -> np.ndarray:
    """
    Computes spectral 2-norm condition number kappa_2(J) = sigma_max / sigma_min for (..., 3, 3) Jacobians.
    """
    orig_shape = J.shape[:-2]
    J_flat = J.reshape(-1, 3, 3)
    s = np.linalg.svd(J_flat, compute_uv=False)  # shape (N, 3), sorted descending
    kappa = s[:, 0] / np.maximum(s[:, 2], 1e-12)
    return kappa.reshape(orig_shape)


def compute_sobolev_h2_norm(coeffs: np.ndarray, freqs: Tuple[int, ...] = (1, 2)) -> float:
    """
    Computes Sobolev H^2 energy norm of Fourier shear stream parameters.
    """
    pi = np.pi
    total_energy = 0.0
    for i, fx in enumerate(freqs):
        for j, fy in enumerate(freqs):
            k2 = (pi ** 2) * (fx ** 2 + fy ** 2)
            c = coeffs[..., i, j, :]
            total_energy += float(np.sum(c ** 2) * (1.0 + k2 + (k2 ** 2)))
    return np.sqrt(total_energy)
