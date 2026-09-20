import jax.numpy as jnp
from tfne_jax.validation.passive_cable import solve_passive_cable_1d, analytic_solution

def test_passive_cable_convergence():
    L = 1.0
    lambda_val = 0.2
    v_0 = 1.0
    v_L = 0.5
    
    results = []
    for n in [50, 100, 200]:
        x, v_num = solve_passive_cable_1d(L, n, lambda_val**2, v_0, v_L)
        v_ana = analytic_solution(x, L, lambda_val, v_0, v_L)
        
        l2_err = jnp.sqrt(jnp.mean((v_num - v_ana)**2))
        max_err = jnp.max(jnp.abs(v_num - v_ana))
        results.append((n, l2_err, max_err))
        
    print(f"\nGrid Refinement Results: {results}")
    
    # Check convergence (last error should be smaller than first)
    assert results[-1][1] < results[0][1], "Convergence failed"
    assert jnp.isfinite(results[-1][1]), "Non-finite error"
    print("PASS")

if __name__ == "__main__":
    test_passive_cable_convergence()
