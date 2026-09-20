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
from tfne_jax.models.cortical_column import ColumnConfig, simulate_cortical_column

def run_test():
    print("Starting Cortical Column Smoke Test (N=20: 16E, 3PV, 1SST)...")
    key = jax.random.PRNGKey(123)
    n_e, n_pv, n_sst = 16, 3, 1
    n = n_e + n_pv + n_sst
    
    # 0=E, 1=PV, 2=SST
    pop_id = jnp.concatenate([
        jnp.zeros(n_e, dtype=jnp.int32),
        jnp.ones(n_pv, dtype=jnp.int32),
        2 * jnp.ones(n_sst, dtype=jnp.int32)
    ])
    
    params_e = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    params_pv = IzhikevichParams(a=0.1, b=0.2, c=-65.0, d=2.0)
    params_sst = IzhikevichParams(a=0.02, b=0.25, c=-65.0, d=2.0)
    
    syn_params = SynapseParams(
        tau_exc=5.0*MS, tau_inh=10.0*MS,
        g_exc=0.1, g_inh=0.5,
        e_exc=0.0*MV, e_inh=-80.0*MV
    )
    
    grid = create_grid_2d(nx=10, ny=10, dx=20.0*UM, dy=20.0*UM)
    
    pos_key, conn_key = jax.random.split(key)
    neuron_pos = jax.random.uniform(pos_key, (n, 2)) * 200.0
    
    p_conn = 0.2
    conn_mat = jax.random.bernoulli(conn_key, p_conn, (n, n))
    is_inhibitory = pop_id > 0
    w_exc = conn_mat * (~is_inhibitory[:, None])
    w_inh = conn_mat * (is_inhibitory[:, None])
    
    config = ColumnConfig(
        params_e=params_e,
        params_pv=params_pv,
        params_sst=params_sst,
        syn_params=syn_params,
        grid=grid,
        neuron_pos=neuron_pos,
        electrode_pos=jnp.array([[100.0, 100.0]]),
        w_exc=w_exc,
        w_inh=w_inh,
        pop_id=pop_id,
        mollifier_eps=20.0*UM,
        sigma_e=0.3,
        dt=0.1*MS,
        delay_steps=1
    )
    
    duration = 50.0*MS
    i_app = jax.random.uniform(key, (n,)) * 10.0 + 5.0
    
    final_state, history = simulate_cortical_column(config, duration, i_app)
    
    print(f"Test Completed.")
    print(f"Total spikes: {jnp.sum(history['spike'])}")
    
    if jnp.all(jnp.isfinite(history['v'])) and jnp.all(jnp.isfinite(history['lfp'])):
        print("PASS: Column simulation stable.")
    else:
        print("FAIL: NaN or Inf detected.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
