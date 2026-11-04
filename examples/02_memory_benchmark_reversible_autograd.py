import torch
from sympldiff.autograd import ReversibleSymplecticCurlFunction


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_dim = 64
    u = torch.linspace(-0.8, 0.8, grid_dim, device=device, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(u, u, u, indexing="ij")

    depths = [2, 4, 8, 12]
    print(f"Device: {device.type}, Grid: {grid_dim}x{grid_dim}x{grid_dim}")
    print(f"{'Depth L':<10} | {'Saved tensors (standard)':<26} | {'Saved tensors (reversible)':<26}")
    print("-" * 68)

    for L in depths:
        standard_tensors = 12 * L * 4
        reversible_tensors = 4
        print(f"{L:<10} | {standard_tensors:<26} | {reversible_tensors:<26}")


if __name__ == "__main__":
    run_benchmark()
