import jax
import jax.numpy as jnp
from typing import NamedTuple
from tfne_jax.core.state import NeuronState, NetworkState
from tfne_jax.physics.izhikevich import IzhikevichParams, compute_izhikevich_step
from tfne_jax.physics.sources import Grid2D, gaussian_mollifier, compute_volumetric_source
from tfne_jax.physics.field import solve_poisson_2d
from tfne_jax.physics.lfp import sample_lfp

class SingleNeuronConfig(NamedTuple):
    izh_params: IzhikevichParams
    grid: Grid2D
    neuron_pos: jnp.ndarray # [1, 2]
    electrode_pos: jnp.ndarray # [M, 2]
    mollifier_eps: float
    sigma_e: float
    dt: float

def simulate_single_neuron_surrounding(
    config: SingleNeuronConfig,
    duration_ms: float,
    i_app_val: float
):
    num_steps = int(duration_ms / config.dt)
    n = 1
    
    # Precompute mollifiers
    mollifiers = gaussian_mollifier(config.grid, config.neuron_pos, config.mollifier_eps)
    
    def step_fn(carry, _):
        net_state = carry
        
        # Drive neuron
        i_app = jnp.array([i_app_val])
        i_syn = jnp.zeros(n) # No synapses in single neuron smoke test
        
        # Update dynamics
        new_neurons = compute_izhikevich_step(
            net_state.neurons,
            config.izh_params,
            i_syn,
            i_app,
            config.dt
        )
        
        # Source proxy: use total current (syn + app - recovery)
        # For smoke test, let's just use i_app + i_syn as a simple proxy
        # In a real model, this would be the sum of all membrane currents.
        i_mem = i_app + i_syn - net_state.neurons.u
        
        # Volumetric source
        source_grid = compute_volumetric_source(i_mem, mollifiers)
        
        # Field solve
        phi_e = solve_poisson_2d(config.grid, source_grid, config.sigma_e)
        
        # Sample LFP
        lfp = sample_lfp(phi_e, config.grid, config.electrode_pos)
        
        new_state = net_state._replace(
            neurons=new_neurons,
            time_step=net_state.time_step + 1
        )
        
        outputs = {
            "v": new_neurons.v,
            "spike": new_neurons.spike,
            "lfp": lfp,
            "i_mem": i_mem
        }
        
        return new_state, outputs

    # Initial state
    init_neurons = NeuronState(
        v=jnp.ones(n) * -65.0,
        u=jnp.ones(n) * -65.0 * config.izh_params.b,
        syn_exc=jnp.zeros(n),
        syn_inh=jnp.zeros(n),
        spike=jnp.zeros(n)
    )
    init_state = NetworkState(
        neurons=init_neurons,
        delay_queue=jnp.zeros((1, n)), # Not used here
        time_step=0
    )
    
    final_state, history = jax.lax.scan(step_fn, init_state, jnp.arange(num_steps))
    
    return final_state, history
