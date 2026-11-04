# sympldiff

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

`sympldiff` is a PyTorch and NumPy library for exact volume-preserving diffeomorphic coordinate transformations and memory-efficient backpropagation, designed for computational anatomy and medical image registration.

The package implements Lie-algebraic coordinate shear splitting on 3D bounded domains $\Omega = [-1, 1]^3$. Because each stage uses planar coordinate shears with triangular Jacobians, volume preservation holds analytically:
$$\det(D\Phi(\mathbf{x})) \equiv 1.000000000000000$$
to machine precision ($|\det(J) - 1| < 10^{-14}$ in double precision) with zero numerical Taylor approximations.

Intermediate activations are not cached during training. The custom autograd operator (`ReversibleSymplecticCurlFunction`) reconstructs coordinates dynamically during backpropagation via closed-form inversion, achieving $\mathcal{O}(1)$ activation memory overhead across arbitrary network depths.

---

## Comparison with Existing Methods

| Method | Volume preservation | Folding ($\det \le 0$) | Inverse computation | Backprop activation memory | Domain confinement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ANTs / SyN | Uncontrolled drift | $< 0.1\%$ | Iterative numerical | N/A (CPU) | Unrestricted |
| VoxelMorph / SVF | Approximate penalty | $0.1 - 2.0\%$ | Scaling-and-squaring | $\mathcal{O}(L \cdot N_{\text{vox}})$ | Unrestricted |
| RealNVP | Density change | Boundary foldings | Closed form | $\mathcal{O}(1)$ | Boundary escape |
| **`sympldiff`** | **Exact machine $\equiv 1.0$** | **Strictly $0.0000\%$** | **Exact closed form** | **$\mathcal{O}(1)$ (Constant)** | **$0.000000$ (Compact window)** |

---

## Installation

```bash
git clone https://github.com/alvaro-garcia-bme/sympldiff.git
cd sympldiff
pip install -e ".[dev]"
```

Or install directly via pip:
```bash
pip install git+https://github.com/alvaro-garcia-bme/sympldiff.git
```

Dependencies: `torch >= 2.0.0`, `numpy >= 1.22.0`, `scipy >= 1.9.0`.

---

## Quickstart

```python
import torch
from sympldiff import SymplecticDiffeo3D, create_normalized_grid_3d

# Initialize volume-preserving diffeomorphic module
transform = SymplecticDiffeo3D(
    steps=4,
    freqs=(1, 2),
    dt=1.0,
    bc_mode="compact_dirichlet"
)

# Coordinates on [-1, 1]^3
X, Y, Z = create_normalized_grid_3d((64, 64, 64))

# Deform a 3D medical volume with O(1) autograd memory
volume = torch.randn(1, 1, 64, 64, 64)
warped = transform.deform_volume(volume)

loss = (warped ** 2).sum()
loss.backward()
```

---

## Mathematical Formulation

The package decomposes the volume-preserving Lie algebra $\mathfrak{sdiff}(\Omega)$ into six directional coordinate shear families:
$$\mathbf{v}^{(1)} = \begin{pmatrix} 0 \\ f_1(x, z) \\ 0 \end{pmatrix}, \quad \mathbf{v}^{(2)} = \begin{pmatrix} 0 \\ 0 \\ f_2(x, y) \end{pmatrix}, \quad \dots, \quad \mathbf{v}^{(6)} = \begin{pmatrix} 0 \\ f_6(z, x) \\ 0 \end{pmatrix}$$

Because each stream potential $f_k$ depends only on the two passive coordinates, convective acceleration vanishes identically:
$$(\mathbf{v}^{(k)} \cdot \nabla)\mathbf{v}^{(k)} \equiv 0$$

This yields:
1. Exact flow integration: $\mathcal{S}_k^{(\Delta t)}(\mathbf{x}) = \mathbf{x} + \Delta t \, \mathbf{v}^{(k)}(\mathbf{x})$.
2. Exact unit determinant: $\det(D\mathcal{S}_k) \equiv 1.000000000000000$.
3. Exact closed-form inverse: $(\mathcal{S}_k^{(\Delta t)})^{-1} \equiv \mathcal{S}_k^{(-\Delta t)}$.

A 12-stage symmetric Strang splitting provides second-order trajectory convergence $\mathcal{O}(\Delta t^2)$.

---

## Tests and Examples

Run unit tests via `pytest`:
```bash
pytest tests/ -v
```

Runnable scripts in `examples/`:
- `examples/01_exact_volume_preservation_demo.py`: Verifies machine-precision determinant and closed-form inversion.
- `examples/02_memory_benchmark_reversible_autograd.py`: Profiles memory scaling between standard and reversible backpropagation.
- `examples/03_3d_registration_quickstart.py`: Pairwise 3D image registration optimization loop.

---

## Citation

```bibtex
@article{sympldiff2026,
  title = {sympldiff: Exact Volume-Preserving Diffeomorphic Registration and Symplectic Shear Splitting with O(1)-Memory Reversible Autograd in PyTorch},
  author = {Garcia, Alvaro and Research Group},
  journal = {Journal of Open Source Software},
  year = {2026},
  doi = {10.21105/joss.xxxxx}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
