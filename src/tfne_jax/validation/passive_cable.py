import jax.numpy as jnp

def solve_passive_cable_1d(L, n, lambda_sq, v_0, v_L):
    """
    Steady-state passive cable equation:
    -d2V/dx2 + (1/lambda^2)V = 0
    """
    dx = L / (n - 1)
    x = jnp.linspace(0, L, n)
    
    # Finite difference matrix for -d2V/dx2 + (1/lambda^2)V
    # Using central difference for d2V/dx2
    diag = (2.0 / dx**2 + 1.0 / lambda_sq) * jnp.ones(n)
    off_diag = (-1.0 / dx**2) * jnp.ones(n - 1)
    
    A = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
    
    # Boundary conditions: V(0) = v_0, V(L) = v_L
    A = A.at[0, :].set(0.0)
    A = A.at[0, 0].set(1.0)
    A = A.at[-1, :].set(0.0)
    A = A.at[-1, -1].set(1.0)
    
    b = jnp.zeros(n)
    b = b.at[0].set(v_0)
    b = b.at[-1].set(v_L)
    
    return x, jnp.linalg.solve(A, b)

def analytic_solution(x, L, lambda_val, v_0, v_L):
    # V(x) = A*exp(-x/lambda) + B*exp(x/lambda)
    # A + B = v_0
    # A*exp(-L/lambda) + B*exp(L/lambda) = v_L
    # B = (v_L - v_0*exp(-L/lambda)) / (exp(L/lambda) - exp(-L/lambda))
    # A = v_0 - B
    
    ex_l = jnp.exp(-L/lambda_val)
    ex_p = jnp.exp(L/lambda_val)
    
    B = (v_L - v_0 * ex_l) / (ex_p - ex_l)
    A = v_0 - B
    
    return A * jnp.exp(-x/lambda_val) + B * jnp.exp(x/lambda_val)
