import torch
import torch.optim as optim
from sympldiff import SymplecticDiffeo3D, create_normalized_grid_3d


def main():
    device = torch.device("cpu")
    dim = 32

    X, Y, Z = create_normalized_grid_3d((dim, dim, dim), device=device)
    radius_sq_fixed = (X / 0.5) ** 2 + (Y / 0.4) ** 2 + (Z / 0.6) ** 2
    fixed_vol = torch.exp(-2.0 * radius_sq_fixed).unsqueeze(0).unsqueeze(0)

    radius_sq_moving = ((X - 0.1) / 0.45) ** 2 + ((Y + 0.1) / 0.45) ** 2 + (Z / 0.55) ** 2
    moving_vol = torch.exp(-2.0 * radius_sq_moving).unsqueeze(0).unsqueeze(0)

    initial_mse = torch.mean((moving_vol - fixed_vol) ** 2).item()
    print(f"Initial MSE: {initial_mse:.6f}")

    reg_module = SymplecticDiffeo3D(
        steps=2,
        freqs=(1, 2),
        dt=1.0,
        bc_mode="compact_dirichlet",
        margin=0.15,
        init_scale=0.01
    ).to(device)

    optimizer = optim.Adam(reg_module.parameters(), lr=0.05)

    for it in range(1, 16):
        optimizer.zero_grad()
        warped_moving = reg_module.deform_volume(moving_vol)
        loss = torch.mean((warped_moving - fixed_vol) ** 2)
        loss.backward()
        optimizer.step()

        if it % 5 == 0 or it == 1:
            print(f"Iter {it:02d}/15 - Loss: {loss.item():.6f}")

    final_mse = loss.item()
    print(f"Final MSE: {final_mse:.6f}")


if __name__ == "__main__":
    main()
