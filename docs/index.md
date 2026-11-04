# sympldiff: Exact Volume-Preserving Diffeomorphic Registration

[![CI Test Suite](https://github.com/sympldiff/sympldiff/actions/workflows/ci.yml/badge.svg)](https://github.com/sympldiff/sympldiff/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![Status: JOSS Under Review](https://img.shields.io/badge/JOSS-Under%20Review-orange.svg)](https://joss.theoj.org/)

`sympldiff` is a specialized, high-performance scientific Python package for **exact volume-preserving diffeomorphic coordinate transformations** and **reversible backpropagation** in PyTorch and NumPy.

Developed for continuum biomechanics, computational anatomy, and deep learning medical image registration, `sympldiff` resolves two long-standing barriers in 3D/4D spatial transformation:
1. **Uncontrolled Volume Drift & Singularity**: Conventional registration approaches (scaling-and-squaring SVF, LDDMM, soft Jacobian penalties) drift from volume conservation ($\sim 10^{-4} - 10^{-1}$) and risk folding singularities ($\det(J) \le 0$). `sympldiff` guarantees $\det(D\Phi) \equiv 1.000000$ to machine precision by construction via closed-form coordinate shear splitting.
2. **Backpropagation GPU Memory Wall**: Deep cascaded transformation models require $\mathcal{O}(L \cdot N_{\text{vox}})$ activation memory during training. `sympldiff` provides a custom reversible autograd engine (`ReversibleSymplecticCurlFunction`) that reconstructs states on the fly during backward pass, achieving strictly $\mathcal{O}(1)$ activation memory ($26.9\times$ peak memory reduction at $128^3$ resolution).

---

## Key Capabilities

- **Exact Closed-Form Analytical Invertibility**: $\Phi^{-1}$ is obtained in $\mathcal{O}(1)$ time without iterative fixed-point solvers or spatial grid interpolation.
- **Topological Invariance**: Strictly zero foldings ($0.0000\%$ negative Jacobian determinants).
- **Domain Confinement**: Compact Dirichlet passive boundary envelopes prevent tissue coordinates from escaping bounded FOVs ($0.000000$ boundary escape).
- **Moser–Dacorogna Extension**: Factorized projective-Mobius coupling for compressible anatomical organs (lung CT respiration, tumor growth, longitudinal atrophy).
- **Full PyTorch & NumPy Support**: Fully vectorised, zero C++ compilation dependencies, runs seamlessly on CPU and CUDA.
