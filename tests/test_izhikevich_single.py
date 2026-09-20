import pytest
import jax.numpy as jnp
from tfne_jax.core.units import MS, MV, UM, PA
from tfne_jax.physics.izhikevich import IzhikevichParams
from tfne_jax.physics.sources import create_grid_2d
from tfne_jax.models.single_neuron import SingleNeuronConfig, simulate_single_neuron_surrounding

def test_izhikevich_single():
    # Setup
    params = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    grid = create_grid_2d(nx=10, ny=10, dx=10.0*UM, dy=10.0*UM)
    config = SingleNeuronConfig(
        izh_params=params,
        grid=grid,
        neuron_pos=jnp.array([[50.0, 50.0]]),
        electrode_pos=jnp.array([[20.0, 20.0], [80.0, 80.0]]),
        mollifier_eps=15.0*UM,
        sigma_e=0.3, # S/m -> nS/um scale? Let's stay consistent
        dt=0.1*MS
    )
    
    # Run
    duration = 50.0*MS
    final_state, history = simulate_single_neuron_surrounding(config, duration, i_app_val=10.0*PA)
    
    # Assert
    assert history["v"].shape == (500, 1)
    assert history["lfp"].shape == (500, 2)
    assert jnp.all(jnp.isfinite(history["v"]))
    assert jnp.all(jnp.isfinite(history["lfp"]))
    
    # Check if spikes occur (with 10 PA it should eventually)
    # 50 ms might be short for 10 PA, but let's check finite values first.
    print(f"Final V: {final_state.neurons.v}")
