import sys
import os
import jax.numpy as jnp

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "repos/gamma/src"))

from tfne_jax.core.units import MS, MV, UM, PA
from tfne_jax.physics.izhikevich import IzhikevichParams
from tfne_jax.physics.sources import create_grid_2d
from tfne_jax.models.single_neuron import SingleNeuronConfig, simulate_single_neuron_surrounding
from tfne_jax.optimization.sgd_fit import fit_single_neuron

def run_test():
    print("Starting SGD Single Neuron Adaptation Smoke Test...")
    params = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    grid = create_grid_2d(nx=10, ny=10, dx=10.0*UM, dy=10.0*UM)
    config = SingleNeuronConfig(
        izh_params=params,
        grid=grid,
        neuron_pos=jnp.array([[50.0, 50.0]]),
        electrode_pos=jnp.array([[50.0, 50.0]]),
        mollifier_eps=15.0*UM,
        sigma_e=0.3,
        dt=0.1*MS
    )
    
    # Generate synthetic target with i_app = 15.0
    duration = 50.0*MS
    _, history_target = simulate_single_neuron_surrounding(config, duration, i_app_val=15.0*PA)
    target_v = history_target["v"].flatten()
    
    # Fit starting from i_app = 10.0 (and other defaults)
    params_fit, losses = fit_single_neuron(target_v, config, duration, num_iters=50, lr=0.1)
    
    print(f"Test Completed.")
    print(f"Initial Loss: {losses[0]}")
    print(f"Final Loss: {losses[-1]}")
    
    if losses[-1] < losses[0]:
        print("PASS: Loss decreased.")
    else:
        print("FAIL: Loss did not decrease.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
