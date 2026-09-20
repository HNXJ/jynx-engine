import jax
import jax.numpy as jnp
from typing import NamedTuple
from tfne_jax.core.state import NeuronState

class SynapseParams(NamedTuple):
    tau_exc: float  # Decay constant for excitatory synapses (ms)
    tau_inh: float  # Decay constant for inhibitory synapses (ms)
    g_exc: float    # Peak excitatory conductance
    g_inh: float    # Peak inhibitory conductance
    e_exc: float    # Excitatory reversal potential (mV)
    e_inh: float    # Inhibitory reversal potential (mV)

def compute_synapse_step(
    state: NeuronState,
    params: SynapseParams,
    delayed_spikes: jnp.ndarray, # Shape [N]
    w_exc: jnp.ndarray,         # Connectivity matrix [N, N] (E-to-all)
    w_inh: jnp.ndarray,         # Connectivity matrix [N, N] (I-to-all)
    dt: float
) -> NeuronState:
    """
    Update synaptic conductance traces and compute synaptic current.
    """
    # Decay traces
    new_syn_exc = state.syn_exc * jnp.exp(-dt / params.tau_exc)
    new_syn_inh = state.syn_inh * jnp.exp(-dt / params.tau_inh)
    
    # Receive spikes
    # W_exc.T @ delayed_spikes gives the input to each neuron from all E neurons
    new_syn_exc += jnp.dot(w_exc.T, delayed_spikes)
    new_syn_inh += jnp.dot(w_inh.T, delayed_spikes)
    
    return state._replace(
        syn_exc=new_syn_exc,
        syn_inh=new_syn_inh
    )

def compute_synaptic_current(
    state: NeuronState,
    params: SynapseParams
) -> jnp.ndarray:
    """
    I_syn = g_exc * syn_exc * (E_exc - V) + g_inh * syn_inh * (E_inh - V)
    """
    i_exc = params.g_exc * state.syn_exc * (params.e_exc - state.v)
    i_inh = params.g_inh * state.syn_inh * (params.e_inh - state.v)
    return i_exc + i_inh
