import jax
import jax.numpy as jnp
from tfne_jax.physics.sources import create_grid_2d

def build_anisotropic_laplacian(grid, sigma_xx, sigma_yy):
    nx, ny = grid.shape
    dx, dy = grid.dx, grid.dy
    
    def d2_1d(m, d):
        diag = -2.0 * jnp.ones(m)
        off = jnp.ones(m - 1)
        L = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
        L = L.at[0, 0].set(-1.0)
        L = L.at[-1, -1].set(-1.0)
        return L / (d**2)

    Dx2 = d2_1d(nx, dx)
    Dy2 = d2_1d(ny, dy)
    Ix = jnp.eye(nx)
    Iy = jnp.eye(ny)
    
    # Laplacian: div(sigma * grad phi) = sigma_xx * d2/dx2 + sigma_yy * d2/dy2
    # Operator for -div(sigma*grad phi)
    return -(sigma_xx * jnp.kron(Dx2, Iy) + sigma_yy * jnp.kron(Ix, Dy2))

def compute_anisotropy_moment(phi, grid):
    # X axis is axis 0, Y axis is axis 1
    # grid.shape is (nx, ny)
    X, Y = jnp.meshgrid(jnp.arange(grid.shape[0]), jnp.arange(grid.shape[1]), indexing='ij')
    x0, y0 = grid.shape[0] / 2, grid.shape[1] / 2
    
    phi2 = phi**2
    total = jnp.sum(phi2)
    # Mxx is spread along X axis (axis 0)
    # Myy is spread along Y axis (axis 1)
    Mxx = jnp.sum(phi2 * (X - x0)**2) / total
    Myy = jnp.sum(phi2 * (Y - y0)**2) / total
    return Mxx / Myy

def create_gaussian_source(grid, sigma=2.0):
    X, Y = jnp.meshgrid(jnp.arange(grid.shape[0]), jnp.arange(grid.shape[1]), indexing='ij')
    x0, y0 = grid.shape[0] / 2, grid.shape[1] / 2
    src = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return src - jnp.mean(src)
