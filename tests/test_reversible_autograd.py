import pytest
import torch
from sympldiff.autograd import ReversibleSymplecticCurlFunction
from sympldiff.operators import eval_sobolev_stream_2d_torch


def _unrolled_standard_forward(x_in, y_in, z_in, coeffs, freqs, dt, bc_mode, strang_symmetric, margin):
    x = x_in.clone()
    y = y_in.clone()
    z = z_in.clone()
    half_dt = 0.5 * dt if strang_symmetric else dt
    schedule = [(s, half_dt) for s in range(6)] + ([(s, half_dt) for s in reversed(range(6))] if strang_symmetric else [])

    for step in range(coeffs.shape[0]):
        c_step = coeffs[step]
        for s_idx, dt_s in schedule:
            if s_idx == 0:
                y = y + dt_s * eval_sobolev_stream_2d_torch(x, z, c_step[0], freqs, bc_mode, 1.0, margin)
            elif s_idx == 1:
                z = z + dt_s * eval_sobolev_stream_2d_torch(x, y, c_step[1], freqs, bc_mode, 1.0, margin)
            elif s_idx == 2:
                z = z + dt_s * eval_sobolev_stream_2d_torch(y, x, c_step[2], freqs, bc_mode, 1.0, margin)
            elif s_idx == 3:
                x = x + dt_s * eval_sobolev_stream_2d_torch(y, z, c_step[3], freqs, bc_mode, 1.0, margin)
            elif s_idx == 4:
                x = x + dt_s * eval_sobolev_stream_2d_torch(z, y, c_step[4], freqs, bc_mode, 1.0, margin)
            elif s_idx == 5:
                y = y + dt_s * eval_sobolev_stream_2d_torch(z, x, c_step[5], freqs, bc_mode, 1.0, margin)
    return x, y, z


def test_reversible_autograd_gradient_equivalence():
    torch.manual_seed(42)
    device = torch.device("cpu")
    dtype = torch.float64

    u = torch.linspace(-0.7, 0.7, 6, dtype=dtype, device=device)
    Z, Y, X = torch.meshgrid(u, u, u, indexing="ij")

    coeffs_rev = (torch.randn(3, 6, 2, 2, 4, dtype=dtype, device=device) * 0.1).requires_grad_(True)
    coeffs_unroll = coeffs_rev.detach().clone().requires_grad_(True)

    freqs = (1, 2)
    dt = 1.0
    bc_mode = "compact_dirichlet"
    margin = 0.15

    x_r, y_r, z_r = ReversibleSymplecticCurlFunction.apply(
        X, Y, Z, coeffs_rev, freqs, dt, bc_mode, (1.0, 1.0, 1.0), True, margin
    )
    loss_r = (x_r ** 2 + y_r ** 2 + z_r ** 2).sum()
    loss_r.backward()

    x_u, y_u, z_u = _unrolled_standard_forward(
        X, Y, Z, coeffs_unroll, freqs, dt, bc_mode, True, margin
    )
    loss_u = (x_u ** 2 + y_u ** 2 + z_u ** 2).sum()
    loss_u.backward()

    max_fwd_diff = max(
        torch.max(torch.abs(x_r - x_u)).item(),
        torch.max(torch.abs(y_r - y_u)).item(),
        torch.max(torch.abs(z_r - z_u)).item(),
    )
    assert max_fwd_diff < 1e-13

    max_grad_diff = torch.max(torch.abs(coeffs_rev.grad - coeffs_unroll.grad)).item()
    assert max_grad_diff < 1e-12
