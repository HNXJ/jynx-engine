import jax.numpy as jnp
from tfne_jax.physics.sources import create_grid_2d, gaussian_mollifier, compute_volumetric_source

def test_source_conservation():
    nx, ny = 20, 20
    grid = create_grid_2d(nx, ny, 0.1, 0.1)
    n_neurons = 5
    pos = jnp.array([[0.5, 0.5], [1.0, 1.0], [1.5, 0.5], [0.5, 1.5], [1.0, 0.5]])
    currents = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mollifiers = gaussian_mollifier(grid, pos, 0.05)
    
    q_grid = compute_volumetric_source(currents, mollifiers)
    
    total_grid_current = jnp.sum(q_grid) * 0.01  # Assuming grid.area is 0.01 based on 0.1*0.1
    total_source_current = jnp.sum(currents)
    
    rel_err = jnp.abs(total_grid_current - total_source_current) / total_source_current
    print(f"Conservation error: {rel_err}")
    assert rel_err < 1e-4

if __name__ == "__main__":
    test_source_conservation()
    print("Test passed!")
