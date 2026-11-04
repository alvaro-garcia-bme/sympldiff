"""
Unit Test: Compact Boundary Confinement (Zero Boundary Escape)
==============================================================
Verifies that under 'compact_dirichlet' boundary mode:
1. Points inside the anatomical ROI [-1+margin, 1-margin]^3 never escape the bounded domain [-1, 1]^3
   under the admissible CFL step size bound (Lemma 2).
2. Points at the domain outer boundary frame experience strictly 0.000000 displacement.
"""

import pytest
import numpy as np
from sympldiff.operators import SymplecticCurlDiffeo3D_NP


def test_zero_boundary_escape_compact_dirichlet():
    rng = np.random.RandomState(99)
    margin = 0.25
    u = np.linspace(-1.0, 1.0, 20)
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")

    op = SymplecticCurlDiffeo3D_NP(
        freqs=(1, 2),
        dt=1.0,
        bc_mode="compact_dirichlet",
        strang_symmetric=True,
        margin=margin
    )

    # 3 cascade steps with CFL-admissible parameters
    coeffs_list = [rng.randn(6, 2, 2, 4) * 0.10 for _ in range(3)]
    X_def, Y_def, Z_def = op.forward(X, Y, Z, coeffs_list)

    # Check 1: ROI points never escape [-1, 1]^3
    roi_mask = (np.abs(X) <= 1.0 - margin) & (np.abs(Y) <= 1.0 - margin) & (np.abs(Z) <= 1.0 - margin)
    max_roi_coord = max(
        np.max(np.abs(X_def[roi_mask])),
        np.max(np.abs(Y_def[roi_mask])),
        np.max(np.abs(Z_def[roi_mask]))
    )
    assert max_roi_coord <= 1.00000000001, f"Domain boundary escaped: max ROI coordinate = {max_roi_coord:.6f} > 1.0"

    # Check 2: Outer frame points on boundary have zero displacement
    frame_mask = (
        (np.abs(X) >= 1.0 - margin).astype(int)
        + (np.abs(Y) >= 1.0 - margin).astype(int)
        + (np.abs(Z) >= 1.0 - margin).astype(int)
    ) >= 2
    frame_disp = max(
        np.max(np.abs(X_def[frame_mask] - X[frame_mask])),
        np.max(np.abs(Y_def[frame_mask] - Y[frame_mask])),
        np.max(np.abs(Z_def[frame_mask] - Z[frame_mask]))
    )
    assert frame_disp == 0.0, f"Outer boundary frame moved by {frame_disp:.6e} != 0.0"
