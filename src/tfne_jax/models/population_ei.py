import jax
import jax.numpy as jnp
from typing import NamedTuple
from tfne_jax.core.state import NeuronState, NetworkState
from tfne_jax.physics.izhikevich import IzhikevichParams, compute_izhikevich_step
from tfne_jax.physics.synapses import SynapseParams, compute_synapse_step, compute_synaptic_current
from tfne_jax.physics.sources import Grid2D, gaussian_mollifier, compute_volumetric_source
from tfne_jax.physics.field import solve_poisson_2d
from tfne_jax.physics.lfp import sample_lfp

class PopulationConfig(NamedTuple):
    izh_params_e: IzhikevichParams
    izh_params_i: IzhikevichParams
    syn_params: SynapseParams
    grid: Grid2D
    neuron_pos: jnp.ndarray # [N, 2]
    electrode_pos: jnp.ndarray # [M, 2]
    w_exc: jnp.ndarray # [N, N]
    w_inh: jnp.ndarray # [N, N]
    mollifier_eps: float
    sigma_e: float
    dt: float
    delay_steps: int
    is_inhibitory: jnp.ndarray # [N] boolean mask

def simulate_population_ei(
    config: PopulationConfig,
    duration_ms: float,
    i_app: jnp.ndarray # [N]
):
    num_steps = int(duration_ms / config.dt)
    n = config.neuron_pos.shape[0]
    
    mollifiers = gaussian_mollifier(config.grid, config.neuron_pos, config.mollifier_eps)
    
    # Combined params for vectorized step
    def expand(val_e, val_i):
        return jnp.where(config.is_inhibitory, val_i, val_e)
    
    a = expand(config.izh_params_e.a, config.izh_params_i.a)
    b = expand(config.izh_params_e.b, config.izh_params_i.b)
    c = expand(config.izh_params_e.c, config.izh_params_i.c)
    d = expand(config.izh_params_e.d, config.izh_params_i.d)
    combined_izh = IzhikevichParams(a=a, b=b, c=c, d=d)
    
    def step_fn(carry, _):
        net_state = carry
        
        delayed_spikes = net_state.delay_queue[0]
        
        updated_syn_state = compute_synapse_step(
            net_state.neurons,
            config.syn_params,
            delayed_spikes,
            config.w_exc,
            config.w_inh,
            config.dt
        )
        i_syn = compute_synaptic_current(updated_syn_state, config.syn_params)
        
        new_neurons = compute_izhikevich_step(
            updated_syn_state,
            combined_izh,
            i_syn,
            i_app,
            config.dt
        )
        
        new_queue = jnp.roll(net_state.delay_queue, -1, axis=0)
        new_queue = new_queue.at[-1].set(new_neurons.spike)
        
        # LFP solve is expensive, maybe skip every few steps if needed? 
        # For now, let's keep it every step for smoke test but reduce grid size.
        i_mem = i_app + i_syn - updated_syn_state.u
        source_grid = compute_volumetric_source(i_mem, mollifiers)
        phi_e = solve_poisson_2d(config.grid, source_grid, config.sigma_e)
        lfp = sample_lfp(phi_e, config.grid, config.electrode_pos)
        
        new_state = net_state._replace(
            neurons=new_neurons,
            delay_queue=new_queue,
            time_step=net_state.time_step + 1
        )
        
        outputs = {
            "v": new_neurons.v,
            "spike": new_neurons.spike,
            "lfp": lfp
        }
        
        return new_state, outputs

    # Initial state
    init_v = expand(-65.0, -65.0)
    init_u = init_v * b
    init_neurons = NeuronState(
        v=init_v,
        u=init_u,
        syn_exc=jnp.zeros(n),
        syn_inh=jnp.zeros(n),
        spike=jnp.zeros(n)
    )
    init_state = NetworkState(
        neurons=init_neurons,
        delay_queue=jnp.zeros((config.delay_steps, n)),
        time_step=0
    )
    
    final_state, history = jax.lax.scan(step_fn, init_state, jnp.arange(num_steps))
    
    return final_state, history
