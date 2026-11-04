import pytest
import numpy as np
from sympldiff.operators import SymplecticCurlDiffeo3D_NP


@pytest.mark.parametrize("bc_mode", ["periodic", "dirichlet", "compact_dirichlet"])
@pytest.mark.parametrize("strang_symmetric", [False, True])
def test_exact_analytical_inverse_recovery(bc_mode, strang_symmetric):
    rng = np.random.RandomState(1337)
    u = np.linspace(-0.85, 0.85, 10)
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")

    op = SymplecticCurlDiffeo3D_NP(
        freqs=(1, 2),
        dt=1.0,
        bc_mode=bc_mode,
        strang_symmetric=strang_symmetric,
        margin=0.15
    )

    coeffs_list = [rng.randn(6, 2, 2, 4) * 0.18 for _ in range(4)]
    X_def, Y_def, Z_def = op.forward(X, Y, Z, coeffs_list)
    X_rec, Y_rec, Z_rec = op.inverse(X_def, Y_def, Z_def, coeffs_list)

    err = max(np.max(np.abs(X_rec - X)), np.max(np.abs(Y_rec - Y)), np.max(np.abs(Z_rec - Z)))
    assert err < 1e-12
