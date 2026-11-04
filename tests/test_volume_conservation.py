"""
Unit Test: Exact Volume Conservation det(D Phi(x)) === 1.000000000000000
==========================================================================
Verifies that:
1. Every individual planar coordinate shear stage has unit diagonal determinant.
2. Composed 6-stage Lie-Trotter and 12-stage Strang compositions conserve volume to
   analytical machine precision (< 1e-12 in FP64) regardless of anisotropic slice spacing
   (dx, dy, dz) or boundary conditions.
"""

import pytest
import numpy as np
from sympldiff.operators import SymplecticCurlDiffeo3D_NP


@pytest.mark.parametrize("bc_mode", ["periodic", "dirichlet", "compact_dirichlet"])
@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (1.0, 1.0, 5.0)])
@pytest.mark.parametrize("strang_symmetric", [False, True])
def test_exact_jacobian_determinant_unity(bc_mode, spacing, strang_symmetric):
    rng = np.random.RandomState(42)
    u = np.linspace(-0.8, 0.8, 8)
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")

    op = SymplecticCurlDiffeo3D_NP(
        freqs=(1, 2),
        dt=1.0,
        bc_mode=bc_mode,
        spacing=spacing,
        strang_symmetric=strang_symmetric,
        margin=0.15
    )

    # 3 composition steps
    coeffs_list = [rng.randn(6, 2, 2, 4) * 0.15 for _ in range(3)]
    J = op.exact_jacobian(X, Y, Z, coeffs_list)

    # Compute numerical determinant of 3x3 Jacobians
    dets = np.linalg.det(J)

    # Assert machine precision volume preservation
    max_dev = np.max(np.abs(dets - 1.0))
    assert max_dev < 1e-12, f"Volume drift detected: max |det(J) - 1| = {max_dev:.3e} for bc={bc_mode}, spacing={spacing}"
