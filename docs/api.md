# API Reference: sympldiff

The `sympldiff` package provides both high-level PyTorch neural network modules and reference NumPy numerical routines.

---

## High-Level PyTorch Modules (`sympldiff.layers`)

### `SymplecticDiffeo3D`
```python
sympldiff.SymplecticDiffeo3D(
    steps=4,
    freqs=(1, 2),
    dt=1.0,
    bc_mode="compact_dirichlet",
    spacing=(1.0, 1.0, 1.0),
    strang_symmetric=True,
    margin=0.15,
    init_scale=0.05
)
```
Differentiable volume-preserving coordinate transformation layer.
- **`forward(x, y, z, custom_coeffs=None)`**: Transforms 3D coordinate grids $(x, y, z) \mapsto (x_{\text{def}}, y_{\text{def}}, z_{\text{def}})$.
- **`deform_volume(volume, grid=None)`**: Warps a 3D medical volume tensor of shape `(B, C, D, H, W)` using trilinear interpolation through the volume-preserving diffeomorphism.

---

## Reversible Autograd Engine (`sympldiff.autograd`)

### `ReversibleSymplecticCurlFunction`
Custom `torch.autograd.Function` implementing $\mathcal{O}(1)$-activation memory backpropagation.
- Caches strictly the terminal coordinates in `ctx.save_for_backward`.
- Reconstructs intermediate spatial tensors on the fly backwards via exact closed-form inversion.
- Gradients match standard unrolled PyTorch autograd to $< 10^{-12}$ in float64 and $< 10^{-6}$ in float32.

---

## NumPy Numerical Operators (`sympldiff.operators`)

### `SymplecticCurlDiffeo3D_NP`
```python
sympldiff.SymplecticCurlDiffeo3D_NP(
    freqs=(1, 2),
    dt=1.0,
    bc_mode="compact_dirichlet",
    spacing=(1.0, 1.0, 1.0),
    strang_symmetric=True,
    margin=0.15
)
```
- **`forward(X, Y, Z, coeffs_list)`**: Evaluates forward deformation.
- **`inverse(X_new, Y_new, Z_new, coeffs_list)`**: Computes exact closed-form inverse $\Phi^{-1}$ in $\mathcal{O}(1)$ time (zero iterative solvers).
- **`exact_jacobian(X, Y, Z, coeffs_list)`**: Evaluates continuous analytical $3 \times 3$ Jacobian matrix $D\Phi(\mathbf{x})$.

---

## Moser Compressibility Extension (`sympldiff.moser`)

### `MoserCompressibleSymplecticDiffeo3D_NP`
Composite diffeomorphism $\Psi = \Phi_{\text{SDiff}} \circ D_\rho$ for compressible anatomical deformations (e.g., 4D lung CT respiration, tumor growth).
- **`forward(X, Y, Z, shear_coeffs, density_coeffs)`**: Returns warped coordinates and exact prescribed density field $\rho(\mathbf{x}) > 0$.
- **`inverse(...)`**: Exact closed-form inverse $\Psi^{-1} = D_\rho^{-1} \circ \Phi_{\text{SDiff}}^{-1}$.
- **`exact_jacobian_determinant(...)`**: Computes chain-rule determinant $\det(D\Psi(\mathbf{x})) \equiv \rho(\mathbf{x})$.
