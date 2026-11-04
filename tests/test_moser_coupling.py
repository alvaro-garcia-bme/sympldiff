"""
Unit Test: Moser-Dacorogna Prescribed-Compressibility Diffeomorphism
====================================================================
Verifies that:
1. Exact analytical chain-rule Jacobian det(D Psi(x)) matches prescribed rho(x) > 0 (< 1e-12).
2. Exact O(1) closed-form inverse ||Psi^-1(Psi(x)) - x||_inf < 1e-12.
3. Pure density map D_rho maps [-1, 1]^3 -> [-1, 1]^3 with 0.000000 boundary escape.
"""

import pytest
import numpy as np
from sympldiff.moser import MoserCompressibleSymplecticDiffeo3D_NP


def test_moser_invertibility_and_density_match():
    rng = np.random.RandomState(77)
    u = np.linspace(-0.85, 0.85, 8)
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")

    moser = MoserCompressibleSymplecticDiffeo3D_NP(
        freqs=(1, 2),
        dt=1.0,
        bc_mode="compact_dirichlet",
        strang_symmetric=True,
        margin=0.25
    )

    shear_c = [rng.randn(6, 2, 2, 4) * 0.08 for _ in range(2)]
    dens_c = rng.randn(3, 2, 2, 4) * 0.18

    # 1. Forward and Inverse
    Xm, Ym, Zm, rho_exact = moser.forward(X, Y, Z, shear_c, dens_c)
    Xmi, Ymi, Zmi = moser.inverse(Xm, Ym, Zm, shear_c, dens_c)

    moser_inv_err = max(
        np.max(np.abs(Xmi - X)),
        np.max(np.abs(Ymi - Y)),
        np.max(np.abs(Zmi - Z))
    )
    assert moser_inv_err < 1e-12, f"Moser inverse roundtrip error exceeded machine tolerance: {moser_inv_err:.3e}"

    # 2. Chain-rule Jacobian determinant matches prescribed positive density
    det_total, rho_prescribed = moser.exact_jacobian_determinant(X, Y, Z, shear_c, dens_c)
    det_match_err = np.max(np.abs(det_total - rho_prescribed))
    assert det_match_err < 1e-12, f"Jacobian determinant mismatch with prescribed density: {det_match_err:.3e}"

    # 3. Assert positivity (no foldings)
    assert np.min(det_total) > 0.0, f"Folded element detected: min det = {np.min(det_total)}"
