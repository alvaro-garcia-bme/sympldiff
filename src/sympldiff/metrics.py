from typing import Tuple
import numpy as np


def compute_finite_difference_jacobian_det_3d(
    X_def: np.ndarray,
    Y_def: np.ndarray,
    Z_def: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    indexing: str = "ij"
) -> np.ndarray:
    dx, dy, dz = voxel_spacing
    if indexing == "ij":
        dPhi_x_dx, dPhi_x_dy, dPhi_x_dz = np.gradient(X_def, dx, dy, dz)
        dPhi_y_dx, dPhi_y_dy, dPhi_y_dz = np.gradient(Y_def, dx, dy, dz)
        dPhi_z_dx, dPhi_z_dy, dPhi_z_dz = np.gradient(Z_def, dx, dy, dz)
    else:
        dPhi_x_dy, dPhi_x_dx, dPhi_x_dz = np.gradient(X_def, dy, dx, dz)
        dPhi_y_dy, dPhi_y_dx, dPhi_y_dz = np.gradient(Y_def, dy, dx, dz)
        dPhi_z_dy, dPhi_z_dx, dPhi_z_dz = np.gradient(Z_def, dy, dx, dz)

    return (
        dPhi_x_dx * (dPhi_y_dy * dPhi_z_dz - dPhi_y_dz * dPhi_z_dy)
        - dPhi_x_dy * (dPhi_y_dx * dPhi_z_dz - dPhi_y_dz * dPhi_z_dx)
        + dPhi_x_dz * (dPhi_y_dx * dPhi_z_dy - dPhi_y_dy * dPhi_z_dx)
    )


def compute_folding_percentage(det_J: np.ndarray) -> float:
    """Percentage of elements with non-positive Jacobian determinant."""
    num_folded = np.sum(det_J <= 0.0)
    return float(100.0 * num_folded / det_J.size)


def compute_jacobian_condition_number(J: np.ndarray) -> np.ndarray:
    orig_shape = J.shape[:-2]
    J_flat = J.reshape(-1, 3, 3)
    s = np.linalg.svd(J_flat, compute_uv=False)
    kappa = s[:, 0] / np.maximum(s[:, 2], 1e-12)
    return kappa.reshape(orig_shape)


def compute_sobolev_h2_norm(coeffs: np.ndarray, freqs: Tuple[int, ...] = (1, 2)) -> float:
    pi = np.pi
    total_energy = 0.0
    for i, fx in enumerate(freqs):
        for j, fy in enumerate(freqs):
            k2 = (pi ** 2) * (fx ** 2 + fy ** 2)
            c = coeffs[..., i, j, :]
            total_energy += float(np.sum(c ** 2) * (1.0 + k2 + (k2 ** 2)))
    return np.sqrt(total_energy)
