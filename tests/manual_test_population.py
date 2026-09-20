import sys
import os
import jax
import jax.numpy as jnp

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "repos/gamma/src"))

from tfne_jax.core.units import MS, MV, UM, PA
from tfne_jax.physics.izhikevich import IzhikevichParams
from tfne_jax.physics.synapses import SynapseParams
from tfne_jax.physics.sources import create_grid_2d
from tfne_jax.models.population_ei import PopulationConfig, simulate_population_ei

def run_test():
    print("Starting Population E80/I20 Smoke Test (Scaled down to N=20)...")
    key = jax.random.PRNGKey(42)
    n_e, n_i = 16, 4
    n = n_e + n_i
    is_inhibitory = jnp.arange(n) >= n_e
    
    params_e = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    params_i = IzhikevichParams(a=0.1, b=0.2, c=-65.0, d=2.0)
    syn_params = SynapseParams(
        tau_exc=5.0*MS, tau_inh=10.0*MS,
        g_exc=0.1, g_inh=0.5,
        e_exc=0.0*MV, e_inh=-80.0*MV
    )
    
    grid = create_grid_2d(nx=10, ny=10, dx=20.0*UM, dy=20.0*UM)
    
    # Random positions in 200x200 um area
    pos_key, conn_key = jax.random.split(key)
    neuron_pos = jax.random.uniform(pos_key, (n, 2)) * 200.0
    
    # Random sparse connectivity
    p_conn = 0.2
    conn_mat = jax.random.bernoulli(conn_key, p_conn, (n, n))
    w_exc = conn_mat * (~is_inhibitory[:, None])
    w_inh = conn_mat * (is_inhibitory[:, None])
    
    config = PopulationConfig(
        izh_params_e=params_e,
        izh_params_i=params_i,
        syn_params=syn_params,
        grid=grid,
        neuron_pos=neuron_pos,
        electrode_pos=jnp.array([[100.0, 100.0]]),
        w_exc=w_exc,
        w_inh=w_inh,
        mollifier_eps=20.0*UM,
        sigma_e=0.3,
        dt=0.1*MS,
        delay_steps=1,
        is_inhibitory=is_inhibitory
    )
    
    # Run
    duration = 50.0*MS
    i_app = jax.random.uniform(key, (n,)) * 10.0 + 5.0 # Random drive
    
    final_state, history = simulate_population_ei(config, duration, i_app)
    
    print(f"Test Completed.")
    print(f"Total spikes: {jnp.sum(history['spike'])}")
    print(f"Average firing rate: {jnp.sum(history['spike']) / (n * duration / 1000.0)} Hz")
    
    if jnp.all(jnp.isfinite(history['v'])) and jnp.all(jnp.isfinite(history['lfp'])):
        print("PASS: Population simulation stable.")
    else:
        print("FAIL: NaN or Inf detected.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
