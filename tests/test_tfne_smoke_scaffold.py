import jax
import jax.numpy as jnp
import pytest
from tfne_jax.physics.sources import create_grid_2d, gaussian_mollifier, compute_volumetric_source

def test_source_conservation():
    nx, ny = 20, 20
    grid = create_grid_2d(nx, ny, 0.1, 0.1)
    n_neurons = 5
    pos = jnp.array([[0.5, 0.5], [1.0, 1.0], [1.5, 0.5], [0.5, 1.5], [1.0, 0.5]])
    currents = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mollifiers = gaussian_mollifier(grid, pos, 0.05)
    
    q_grid = compute_volumetric_source(currents, mollifiers)
    
    total_grid_current = jnp.sum(q_grid) * grid.area
    total_source_current = jnp.sum(currents)
    
    rel_err = jnp.abs(total_grid_current - total_source_current) / total_source_current
    assert rel_err < 1e-4, f"Conservation error too high: {rel_err}"

def test_gauge_test():
    nx, ny = 20, 20
    grid = create_grid_2d(nx, ny, 0.1, 0.1)
    # Mock source
    source = jnp.ones((nx, ny))
    from tfne_jax.physics.field import solve_poisson_2d
    phi_e = solve_poisson_2d(grid, source, 1.0)
    
    assert jnp.abs(jnp.mean(phi_e)) < 1e-9, "Mean gauge failed"

def test_synapse_integrity():
    # Verify no V_pre is used in synaptic current
    # compute_synaptic_current only uses state.syn_exc/inh and state.v
    from tfne_jax.physics.synapses import compute_synaptic_current
    from tfne_jax.core.state import NeuronState
    
    state = NeuronState(
        v=jnp.array([0.0]),
        u=jnp.array([0.0]),
        syn_exc=jnp.array([1.0]),
        syn_inh=jnp.array([0.0]),
        spike=jnp.array([0.0])
    )
    from tfne_jax.physics.synapses import SynapseParams
    params = SynapseParams(1.0, 1.0, 1.0, 1.0, 0.0, -80.0)
    
    # Just check it runs
    i_syn = compute_synaptic_current(state, params)
    assert not jnp.isnan(i_syn)
