import numpy as np
from sympldiff import SymplecticCurlDiffeo3D_NP, compute_finite_difference_jacobian_det_3d, compute_folding_percentage


def main():
    grid_size = 32
    u = np.linspace(-0.85, 0.85, grid_size)
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")

    operator = SymplecticCurlDiffeo3D_NP(
        freqs=(1, 2),
        dt=1.0,
        bc_mode="compact_dirichlet",
        spacing=(1.0, 1.0, 1.0),
        strang_symmetric=True,
        margin=0.15
    )

    rng = np.random.RandomState(42)
    steps = 4
    coeffs = [rng.randn(6, 2, 2, 4) * 0.15 for _ in range(steps)]

    X_def, Y_def, Z_def = operator.forward(X, Y, Z, coeffs)

    J = operator.exact_jacobian(X, Y, Z, coeffs)
    analytical_dets = np.linalg.det(J)
    max_dev = np.max(np.abs(analytical_dets - 1.0))

    fd_dets = compute_finite_difference_jacobian_det_3d(X_def, Y_def, Z_def, (u[1] - u[0], u[1] - u[0], u[1] - u[0]))
    folding_pct = compute_folding_percentage(fd_dets)

    X_rec, Y_rec, Z_rec = operator.inverse(X_def, Y_def, Z_def, coeffs)
    roundtrip_err = max(np.max(np.abs(X_rec - X)), np.max(np.abs(Y_rec - Y)), np.max(np.abs(Z_rec - Z)))

    print(f"Grid size: {grid_size}^3")
    print(f"Max analytical |det(J) - 1|: {max_dev:.3e}")
    print(f"Numerical folding rate: {folding_pct:.4f}%")
    print(f"Inverse recovery error: {roundtrip_err:.3e}")


if __name__ == "__main__":
    main()
