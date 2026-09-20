import jax
import jax.numpy as jnp
from tfne_jax.physics.sources import Grid2D

def sample_lfp(
    phi_e: jnp.ndarray,     # [X, Y]
    grid: Grid2D,
    electrode_pos: jnp.ndarray # [M, 2]
) -> jnp.ndarray:           # [M]
    """
    Sample extracellular potential at electrode coordinates using nearest neighbor (simple for smoke test).
    """
    # grid.coords: [X, Y, 2]
    # electrode_pos: [M, 2]
    
    # Simple lookup for now
    ix = jnp.clip((electrode_pos[:, 0] / grid.dx).astype(jnp.int32), 0, grid.shape[0] - 1)
    iy = jnp.clip((electrode_pos[:, 1] / grid.dy).astype(jnp.int32), 0, grid.shape[1] - 1)
    
    return phi_e[ix, iy]
