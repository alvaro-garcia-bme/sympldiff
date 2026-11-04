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

    Xm, Ym, Zm, rho_exact = moser.forward(X, Y, Z, shear_c, dens_c)
    Xmi, Ymi, Zmi = moser.inverse(Xm, Ym, Zm, shear_c, dens_c)

    moser_inv_err = max(
        np.max(np.abs(Xmi - X)),
        np.max(np.abs(Ymi - Y)),
        np.max(np.abs(Zmi - Z))
    )
    assert moser_inv_err < 1e-12

    det_total, rho_prescribed = moser.exact_jacobian_determinant(X, Y, Z, shear_c, dens_c)
    det_match_err = np.max(np.abs(det_total - rho_prescribed))
    assert det_match_err < 1e-12
    assert np.min(det_total) > 0.0
