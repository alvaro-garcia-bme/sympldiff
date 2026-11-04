__version__ = "0.1.0"
__author__ = "Alvaro Garcia"
__license__ = "MIT"

from .operators import (
    SymplecticCurlDiffeo3D_NP,
    eval_sobolev_stream_2d_np,
    eval_sobolev_stream_2d_torch,
)
from .autograd import ReversibleSymplecticCurlFunction
from .layers import SymplecticDiffeo3D, create_normalized_grid_3d
from .boundary import boundary_envelope_np, boundary_envelope_torch
from .moser import MoserCompressibleSymplecticDiffeo3D_NP
from .metrics import (
    compute_finite_difference_jacobian_det_3d,
    compute_folding_percentage,
    compute_jacobian_condition_number,
    compute_sobolev_h2_norm,
)

__all__ = [
    "__version__",
    "SymplecticDiffeo3D",
    "ReversibleSymplecticCurlFunction",
    "SymplecticCurlDiffeo3D_NP",
    "MoserCompressibleSymplecticDiffeo3D_NP",
    "create_normalized_grid_3d",
    "boundary_envelope_np",
    "boundary_envelope_torch",
    "eval_sobolev_stream_2d_np",
    "eval_sobolev_stream_2d_torch",
    "compute_finite_difference_jacobian_det_3d",
    "compute_folding_percentage",
    "compute_jacobian_condition_number",
    "compute_sobolev_h2_norm",
]
