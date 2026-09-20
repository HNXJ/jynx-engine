import jax
import jax.numpy as jnp
from tfne_jax.physics.sources import Grid2D

def build_laplacian_2d(grid: Grid2D):
    nx, ny = grid.shape
    
    def get_laplacian_1d(m, d):
        main_diag = -2.0 * jnp.ones(m)
        off_diag = jnp.ones(m - 1)
        L = jnp.diag(main_diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        L = L.at[0, 0].set(-1.0)
        L = L.at[-1, -1].set(-1.0)
        return L / (d**2)

    Lx = get_laplacian_1d(nx, grid.dx)
    Ly = get_laplacian_1d(ny, grid.dy)
    Ix = jnp.eye(nx)
    Iy = jnp.eye(ny)
    return jnp.kron(Lx, Iy) + jnp.kron(Ix, Ly)

def solve_poisson_2d(grid: Grid2D, source: jnp.ndarray, sigma_e: float) -> jnp.ndarray:
    """
    PATH A: Pure Neumann Poisson. 
    Operator: L * phi = b_projected
    RHS must be projected to mean-zero.
    """
    L = sigma_e * build_laplacian_2d(grid)
    b = source.flatten()
    
    # Project RHS to mean-zero
    b_projected = b - jnp.mean(b)
    
    # lstsq solves singular system (phi + const is also a solution)
    phi_flat, _, _, _ = jnp.linalg.lstsq(L, b_projected)
    
    # Reshape and apply mean-zero gauge (phi - mean(phi))
    phi = phi_flat.reshape(grid.shape)
    return phi - jnp.mean(phi)

def compute_operator_residual(grid: Grid2D, phi: jnp.ndarray, source: jnp.ndarray, sigma_e: float) -> float:
    L = sigma_e * build_laplacian_2d(grid)
    b = source.flatten()
    b_projected = b - jnp.mean(b)
    
    phi_flat = phi.flatten()
    residual = jnp.linalg.norm(jnp.dot(L, phi_flat) - b_projected)
    return residual / (jnp.linalg.norm(b_projected) + 1e-12)
