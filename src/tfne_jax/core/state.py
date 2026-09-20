from typing import NamedTuple
import jax.numpy as jnp

class NeuronState(NamedTuple):
    """
    Izhikevich-like point-neuron state.
    Shapes are [N] where N is number of neurons.
    """
    v: jnp.ndarray  # Membrane voltage (mV)
    u: jnp.ndarray  # Recovery variable
    syn_exc: jnp.ndarray  # Excitatory synaptic state (trace)
    syn_inh: jnp.ndarray  # Inhibitory synaptic state (trace)
    spike: jnp.ndarray    # Boolean/float spike events

class NetworkState(NamedTuple):
    """
    Full network state including delay queue.
    """
    neurons: NeuronState
    delay_queue: jnp.ndarray  # Shape [delay_steps, N]
    time_step: int
