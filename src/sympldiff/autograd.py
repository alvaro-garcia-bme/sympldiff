from typing import Tuple, List
import torch
from .operators import eval_sobolev_stream_2d_torch


def _get_stage_schedule(strang_symmetric: bool, dt: float) -> List[Tuple[int, float]]:
    if strang_symmetric:
        half_dt = 0.5 * dt
        forward_seq = [(s, half_dt) for s in range(6)]
        backward_seq = [(s, half_dt) for s in reversed(range(6))]
        return forward_seq + backward_seq
    return [(s, dt) for s in range(6)]


class ReversibleSymplecticCurlFunction(torch.autograd.Function):
    """Custom autograd function reconstructing coordinate states dynamically in backward pass."""

    @staticmethod
    def forward(
        ctx,
        x_in: torch.Tensor,
        y_in: torch.Tensor,
        z_in: torch.Tensor,
        coeffs: torch.Tensor,
        freqs: Tuple[int, ...] = (1, 2),
        dt: float = 1.0,
        bc_mode: str = "compact_dirichlet",
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        strang_symmetric: bool = True,
        margin: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        sx, sy, sz = [float(s) for s in spacing]
        s_mean = (sx + sy + sz) / 3.0
        aniso_scales = (
            s_mean / sy,
            s_mean / sz,
            s_mean / sz,
            s_mean / sx,
            s_mean / sx,
            s_mean / sy,
        )

        x = x_in.clone()
        y = y_in.clone()
        z = z_in.clone()
        schedule = _get_stage_schedule(strang_symmetric, dt)

        with torch.no_grad():
            for step in range(coeffs.shape[0]):
                c_step = coeffs[step]
                for s_idx, dt_s in schedule:
                    sc = aniso_scales[s_idx]
                    if s_idx == 0:
                        y = y + dt_s * eval_sobolev_stream_2d_torch(x, z, c_step[0], freqs, bc_mode, sc, margin)
                    elif s_idx == 1:
                        z = z + dt_s * eval_sobolev_stream_2d_torch(x, y, c_step[1], freqs, bc_mode, sc, margin)
                    elif s_idx == 2:
                        z = z + dt_s * eval_sobolev_stream_2d_torch(y, x, c_step[2], freqs, bc_mode, sc, margin)
                    elif s_idx == 3:
                        x = x + dt_s * eval_sobolev_stream_2d_torch(y, z, c_step[3], freqs, bc_mode, sc, margin)
                    elif s_idx == 4:
                        x = x + dt_s * eval_sobolev_stream_2d_torch(z, y, c_step[4], freqs, bc_mode, sc, margin)
                    elif s_idx == 5:
                        y = y + dt_s * eval_sobolev_stream_2d_torch(z, x, c_step[5], freqs, bc_mode, sc, margin)

        ctx.save_for_backward(x, y, z, coeffs)
        ctx.freqs = freqs
        ctx.dt = dt
        ctx.bc_mode = bc_mode
        ctx.aniso_scales = aniso_scales
        ctx.strang_symmetric = strang_symmetric
        ctx.margin = margin
        return x, y, z

    @staticmethod
    def backward(ctx, grad_x, grad_y, grad_z):
        x_curr, y_curr, z_curr, coeffs = ctx.saved_tensors
        freqs = ctx.freqs
        dt = ctx.dt
        bc_mode = ctx.bc_mode
        aniso_scales = ctx.aniso_scales
        strang_symmetric = ctx.strang_symmetric
        margin = ctx.margin
        schedule = _get_stage_schedule(strang_symmetric, dt)

        x = x_curr.detach().clone()
        y = y_curr.detach().clone()
        z = z_curr.detach().clone()
        gx = grad_x.detach().clone()
        gy = grad_y.detach().clone()
        gz = grad_z.detach().clone()
        grad_coeffs = torch.zeros_like(coeffs)

        for step in reversed(range(coeffs.shape[0])):
            for s_idx, dt_s in reversed(schedule):
                sc = aniso_scales[s_idx]
                c_param = coeffs[step, s_idx].detach().requires_grad_(True)

                if s_idx == 0:
                    u_in = x.detach().requires_grad_(True)
                    v_in = z.detach().requires_grad_(True)
                    with torch.enable_grad():
                        f_val = eval_sobolev_stream_2d_torch(u_in, v_in, c_param, freqs, bc_mode, sc, margin)
                    with torch.no_grad():
                        y = y - dt_s * f_val.detach()
                    du, dv, dc = torch.autograd.grad(f_val, (u_in, v_in, c_param), grad_outputs=gy * dt_s)
                    gx = gx + du
                    gz = gz + dv
                    grad_coeffs[step, s_idx] += dc

                elif s_idx == 1:
                    u_in = x.detach().requires_grad_(True)
                    v_in = y.detach().requires_grad_(True)
                    with torch.enable_grad():
                        f_val = eval_sobolev_stream_2d_torch(u_in, v_in, c_param, freqs, bc_mode, sc, margin)
                    with torch.no_grad():
                        z = z - dt_s * f_val.detach()
                    du, dv, dc = torch.autograd.grad(f_val, (u_in, v_in, c_param), grad_outputs=gz * dt_s)
                    gx = gx + du
                    gy = gy + dv
                    grad_coeffs[step, s_idx] += dc

                elif s_idx == 2:
                    u_in = y.detach().requires_grad_(True)
                    v_in = x.detach().requires_grad_(True)
                    with torch.enable_grad():
                        f_val = eval_sobolev_stream_2d_torch(u_in, v_in, c_param, freqs, bc_mode, sc, margin)
                    with torch.no_grad():
                        z = z - dt_s * f_val.detach()
                    du, dv, dc = torch.autograd.grad(f_val, (u_in, v_in, c_param), grad_outputs=gz * dt_s)
                    gy = gy + du
                    gx = gx + dv
                    grad_coeffs[step, s_idx] += dc

                elif s_idx == 3:
                    u_in = y.detach().requires_grad_(True)
                    v_in = z.detach().requires_grad_(True)
                    with torch.enable_grad():
                        f_val = eval_sobolev_stream_2d_torch(u_in, v_in, c_param, freqs, bc_mode, sc, margin)
                    with torch.no_grad():
                        x = x - dt_s * f_val.detach()
                    du, dv, dc = torch.autograd.grad(f_val, (u_in, v_in, c_param), grad_outputs=gx * dt_s)
                    gy = gy + du
                    gz = gz + dv
                    grad_coeffs[step, s_idx] += dc

                elif s_idx == 4:
                    u_in = z.detach().requires_grad_(True)
                    v_in = y.detach().requires_grad_(True)
                    with torch.enable_grad():
                        f_val = eval_sobolev_stream_2d_torch(u_in, v_in, c_param, freqs, bc_mode, sc, margin)
                    with torch.no_grad():
                        x = x - dt_s * f_val.detach()
                    du, dv, dc = torch.autograd.grad(f_val, (u_in, v_in, c_param), grad_outputs=gx * dt_s)
                    gz = gz + du
                    gy = gy + dv
                    grad_coeffs[step, s_idx] += dc

                elif s_idx == 5:
                    u_in = z.detach().requires_grad_(True)
                    v_in = x.detach().requires_grad_(True)
                    with torch.enable_grad():
                        f_val = eval_sobolev_stream_2d_torch(u_in, v_in, c_param, freqs, bc_mode, sc, margin)
                    with torch.no_grad():
                        y = y - dt_s * f_val.detach()
                    du, dv, dc = torch.autograd.grad(f_val, (u_in, v_in, c_param), grad_outputs=gy * dt_s)
                    du_val = du if du is not None else 0.0
                    dv_val = dv if dv is not None else 0.0
                    gz = gz + du_val
                    gx = gx + dv_val
                    grad_coeffs[step, s_idx] += dc

        return gx, gy, gz, grad_coeffs, None, None, None, None, None, None
