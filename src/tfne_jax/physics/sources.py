import jax
import jax.numpy as jnp
from typing import NamedTuple

class Grid2D(NamedTuple):
    coords: jnp.ndarray  # Shape [X, Y, 2]
    dx: float
    dy: float
    area: float
    shape: tuple

def create_grid_2d(nx: int, ny: int, dx: float, dy: float) -> Grid2D:
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dy
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    coords = jnp.stack([X, Y], axis=-1)
    return Grid2D(
        coords=coords,
        dx=dx,
        dy=dy,
        area=dx * dy,
        shape=(nx, ny)
    )

def gaussian_mollifier(
    grid: Grid2D,
    pos: jnp.ndarray,  # [N, 2]
    eps: float
) -> jnp.ndarray:     # [N, X, Y]
    """
    Compute normalized Gaussian kernels for each neuron.
    """
    # grid.coords: [X, Y, 2]
    # pos: [N, 2]
    
    # Broadcast to [N, X, Y, 2]
    diff = grid.coords[None, :, :, :] - pos[:, None, None, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    
    # Kernel
    eta = jnp.exp(-dist_sq / (2.0 * eps**2))
    
    # Normalize such that sum(eta * area) = 1
    total_area = jnp.sum(eta, axis=(1, 2)) * grid.area
    eta = eta / total_area[:, None, None]
    
    return eta

def compute_volumetric_source(
    currents: jnp.ndarray,  # [N]
    mollifiers: jnp.ndarray # [N, X, Y]
) -> jnp.ndarray:           # [X, Y]
    """
    q(x,t) = sum_n I_n(t) * eta_n(x)
    """
    # [N, 1, 1] * [N, X, Y] -> [N, X, Y]
    # Then sum over N -> [X, Y]
    return jnp.sum(currents[:, None, None] * mollifiers, axis=0)
