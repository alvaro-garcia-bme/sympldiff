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

    coeffs_list = [rng.randn(6, 2, 2, 4) * 0.15 for _ in range(3)]
    J = op.exact_jacobian(X, Y, Z, coeffs_list)
    dets = np.linalg.det(J)

    max_dev = np.max(np.abs(dets - 1.0))
    assert max_dev < 1e-12
