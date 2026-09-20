import jax
import jax.numpy as jnp
from typing import NamedTuple
from tfne_jax.core.state import NeuronState

class IzhikevichParams(NamedTuple):
    """
    Izhikevich neuron parameters.
    Default values are for typical excitatory neurons.
    """
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    threshold: float = 30.0

def compute_izhikevich_step(
    state: NeuronState,
    params: IzhikevichParams,
    i_syn: jnp.ndarray,
    i_app: jnp.ndarray,
    dt: float
) -> NeuronState:
    """
    Compute one step of Izhikevich dynamics using explicit Euler.
    """
    v = state.v
    u = state.u
    
    # dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    dv = (0.04 * v**2 + 5.0 * v + 140.0 - u + i_syn + i_app) * dt
    # du/dt = a * (b * v - u)
    du = (params.a * (params.b * v - u)) * dt
    
    new_v = v + dv
    new_u = u + du
    
    # Check for spikes
    spike = new_v >= params.threshold
    
    # Reset
    res_v = jnp.where(spike, params.c, new_v)
    res_u = jnp.where(spike, new_u + params.d, new_u)
    
    return state._replace(
        v=res_v,
        u=res_u,
        spike=spike.astype(jnp.float32)
    )
