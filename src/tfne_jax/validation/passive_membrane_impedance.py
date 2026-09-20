import jax.numpy as jnp

def passive_membrane_impedance(f, Cm, gL):
    omega = 2 * jnp.pi * f
    return 1.0 / (gL + 1j * omega * Cm)

def estimate_impedance(f, Cm, gL, dt=0.0001, duration=1.0):
    t = jnp.arange(0, duration, dt)
    I = 1e-9 * jnp.sin(2 * jnp.pi * f * t)
    
    v = jnp.zeros_like(t)
    for i in range(1, len(t)):
        dv = (I[i-1] - gL * v[i-1]) / Cm
        v = v.at[i].set(v[i-1] + dv * dt)
    
    start_idx = int(len(t) * 0.5)
    v_steady = v[start_idx:]
    t_steady = t[start_idx:]
    I_steady = I[start_idx:]
    
    # Projection for complex amplitude
    # V(t) = |Z| I_amp sin(2*pi*f*t + phase)
    # V(t) = (Z_real * sin(wt) + Z_imag * cos(wt)) * I_amp
    
    sin_base = jnp.sin(2 * jnp.pi * f * t_steady)
    cos_base = jnp.cos(2 * jnp.pi * f * t_steady)
    
    v_sin = jnp.dot(v_steady, sin_base) / jnp.dot(sin_base, sin_base)
    v_cos = jnp.dot(v_steady, cos_base) / jnp.dot(cos_base, cos_base)
    
    # V_complex = v_sin - 1j * v_cos (referenced to sin(wt))
    # I_complex = I_amp + 0j
    v_complex = v_sin + 1j * v_cos
    z_complex = v_complex / 1e-9
    
    return jnp.abs(z_complex), jnp.angle(z_complex)
