import sys
import os
import jax.numpy as jnp

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "repos/gamma/src"))

from tfne_jax.core.units import MS, MV, UM, PA
from tfne_jax.physics.izhikevich import IzhikevichParams
from tfne_jax.physics.sources import create_grid_2d
from tfne_jax.models.single_neuron import SingleNeuronConfig, simulate_single_neuron_surrounding

def run_test():
    print("Starting Single Neuron Smoke Test...")
    params = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    grid = create_grid_2d(nx=10, ny=10, dx=10.0*UM, dy=10.0*UM)
    config = SingleNeuronConfig(
        izh_params=params,
        grid=grid,
        neuron_pos=jnp.array([[50.0, 50.0]]),
        electrode_pos=jnp.array([[20.0, 20.0], [80.0, 80.0]]),
        mollifier_eps=15.0*UM,
        sigma_e=0.3,
        dt=0.1*MS
    )
    
    duration = 50.0*MS
    final_state, history = simulate_single_neuron_surrounding(config, duration, i_app_val=10.0*PA)
    
    print(f"Test Completed.")
    print(f"History 'v' shape: {history['v'].shape}")
    print(f"History 'lfp' shape: {history['lfp'].shape}")
    print(f"Final V: {final_state.neurons.v}")
    
    # Check for NaN
    if jnp.any(jnp.isnan(history["v"])) or jnp.any(jnp.isnan(history["lfp"])):
        print("FAIL: NaN detected in outputs")
        sys.exit(1)
    
    print("PASS: All values finite.")

if __name__ == "__main__":
    run_test()
