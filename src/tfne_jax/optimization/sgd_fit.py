import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple
from tfne_jax.core.state import NeuronState, NetworkState
from tfne_jax.physics.izhikevich import IzhikevichParams, compute_izhikevich_step

def smooth_spike_surrogate(v, threshold, delta_v=1.0):
    return jax.nn.sigmoid((v - threshold) / delta_v)

def loss_fn(params_vector, target_v, config, duration_ms):
    """
    params_vector: [a, b, c, d, i_app]
    target_v: [T]
    """
    # Map vector to physical params with constraints
    # a, b, d > 0; c < 0
    a = jax.nn.softplus(params_vector[0])
    b = jax.nn.softplus(params_vector[1])
    c = -jax.nn.softplus(params_vector[2])
    d = jax.nn.softplus(params_vector[3])
    i_app_val = params_vector[4]
    
    params = IzhikevichParams(a=a, b=b, c=c, d=d)
    n = 1
    num_steps = int(duration_ms / config.dt)
    
    def step_fn(carry, _):
        state = carry
        i_app = jnp.array([i_app_val])
        i_syn = jnp.zeros(n)
        
        new_state = compute_izhikevich_step(
            state,
            params,
            i_syn,
            i_app,
            config.dt
        )
        return new_state, new_state.v
    
    init_neurons = NeuronState(
        v=jnp.array([-65.0]),
        u=jnp.array([-65.0 * b]),
        syn_exc=jnp.zeros(n),
        syn_inh=jnp.zeros(n),
        spike=jnp.zeros(n)
    )
    
    _, v_history = jax.lax.scan(step_fn, init_neurons, jnp.arange(num_steps))
    
    # MSE loss on voltage trace
    return jnp.mean((v_history.flatten() - target_v)**2)

def fit_single_neuron(target_v, config, duration_ms, num_iters=100, lr=0.1):
    # Initial guess: [a, b, c, d, i_app]
    # We use log-space or inverse softplus but for smoke test just start near defaults
    init_params = jnp.array([0.02, 0.2, 65.0, 8.0, 10.0])
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, target_v, config, duration_ms)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    params = init_params
    losses = []
    for i in range(num_iters):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        if i % 20 == 0:
            print(f"Iter {i}, Loss: {loss}")
            
    return params, losses
