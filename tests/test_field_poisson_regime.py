import jax.numpy as jnp
from tfne_jax.physics.sources import create_grid_2d
from tfne_jax.physics.field import solve_poisson_2d, compute_operator_residual

def test_poisson_regime():
    grid = create_grid_2d(20, 20, 0.1, 0.1)
    
    # A. Zero source
    source_zero = jnp.zeros((20, 20))
    phi_zero = solve_poisson_2d(grid, source_zero, 1.0)
    assert jnp.allclose(phi_zero, 0.0), f"Zero source failed: {phi_zero.mean()}"
    
    # B. Mean-zero source
    source_mz = jnp.ones((20, 20))
    source_mz = source_mz.at[0,0].set(-399.0)
    phi_mz = solve_poisson_2d(grid, source_mz, 1.0)
    res_mz = compute_operator_residual(grid, phi_mz, source_mz, 1.0)
    assert res_mz < 1e-4, f"Residual too high: {res_mz}"
    
    # C. Constant source 
    source_const = jnp.ones((20, 20))
    phi_c = solve_poisson_2d(grid, source_const, 1.0)
    assert jnp.allclose(phi_c, 0.0), "Constant source should project to zero"
    
    # D. Finite output
    assert jnp.all(jnp.isfinite(phi_mz)), "Non-finite output"
    print(f"Residual: {res_mz:.2e}")
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_poisson_regime()
