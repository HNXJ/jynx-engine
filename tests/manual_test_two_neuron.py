import sys
import os
import jax.numpy as jnp

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "repos/gamma/src"))

from tfne_jax.core.units import MS, MV, UM, PA
from tfne_jax.physics.izhikevich import IzhikevichParams
from tfne_jax.physics.synapses import SynapseParams
from tfne_jax.physics.sources import create_grid_2d
from tfne_jax.models.two_neuron_ei import TwoNeuronConfig, simulate_two_neuron_ei

def run_test():
    print("Starting Two Neuron E-I Smoke Test...")
    params_e = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    params_i = IzhikevichParams(a=0.1, b=0.2, c=-65.0, d=2.0)
    syn_params = SynapseParams(
        tau_exc=5.0*MS, tau_inh=10.0*MS,
        g_exc=1.0, g_inh=1.0,
        e_exc=0.0*MV, e_inh=-80.0*MV
    )
    grid = create_grid_2d(nx=10, ny=10, dx=10.0*UM, dy=10.0*UM)
    
    # E at (30, 50), I at (70, 50)
    neuron_pos = jnp.array([[30.0, 50.0], [70.0, 50.0]])
    
    # W_exc: E (index 0) connects to I (index 1)
    w_exc = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    # W_inh: I (index 1) connects to E (index 0)
    w_inh = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    
    config = TwoNeuronConfig(
        izh_params_e=params_e,
        izh_params_i=params_i,
        syn_params=syn_params,
        grid=grid,
        neuron_pos=neuron_pos,
        electrode_pos=jnp.array([[50.0, 50.0]]),
        w_exc=w_exc,
        w_inh=w_inh,
        mollifier_eps=15.0*UM,
        sigma_e=0.3,
        dt=0.1*MS,
        delay_steps=10 # 1.0 ms delay
    )
    
    # Run with strong drive to E to force spikes
    duration = 100.0*MS
    final_state, history = simulate_two_neuron_ei(config, duration, i_app_e=15.0*PA, i_app_i=0.0*PA)
    
    print(f"Test Completed.")
    print(f"E spikes: {jnp.sum(history['spike'][:, 0])}")
    print(f"I spikes: {jnp.sum(history['spike'][:, 1])}")
    
    # Verify synaptic transmission
    # If E spikes, syn_exc should be non-zero at some point
    max_syn_exc_i = jnp.max(history['syn_exc'][:, 1])
    print(f"Max syn_exc in I: {max_syn_exc_i}")
    
    if max_syn_exc_i > 0:
        print("PASS: Synaptic transmission detected.")
    else:
        print("FAIL: No synaptic transmission detected despite E spikes.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
