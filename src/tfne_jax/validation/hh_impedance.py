import jax
import jax.numpy as jnp

def passive_membrane_impedance(f, g_L, C_m):
    omega = 2 * jnp.pi * f
    return 1.0 / (g_L + 1j * omega * C_m)

def simulate_membrane_response(g_L, C_m, I_stim, dt, duration):
    # dv/dt = (I - g_L*v) / C_m
    steps = int(duration / dt)
    v = jnp.zeros(steps)
    for i in range(steps - 1):
        dv = (I_stim[i] - g_L * v[i]) / C_m
        v = v.at[i+1].set(v[i] + dv * dt)
    return v
