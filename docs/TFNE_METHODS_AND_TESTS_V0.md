# Tensor Field Neural Equation (TFNE) Implementation: Methods and Smoke Tests (V0)

## Overview
This document describes the initial JAX-native implementation of the Tensor Field Neural Equation (TFNE) forward-modeling ladder. The goal is to provide a numerically stable, differentiable scaffold for linking point-neuron membrane dynamics to extracellular potentials (LFP/CSD).

## Scientific Doctrine
- **Forward Modeling**: TFNE maps membrane current proxies to current density and extracellular fields.
- **Mollification**: Singular point sources are replaced by Gaussian kernels $\eta_\epsilon$ to ensure numerical stability and conservation.
- **Poisson Solver**: Solves $\nabla \cdot (\sigma_e \nabla \phi_e) = q$ with mean-zero gauge fixing.
- **Synaptic Rule**: Discrete spike events are transmitted via a delay queue to postsynaptic conductance-based synapses.

## Implementation Ladder

### Stage A: Single Neuron & Surrounding
- **Dynamics**: Izhikevich point-neuron.
- **Source**: Membrane current proxy $I_{mem\_proxy}(t) = I_{app} + I_{syn} - u$.
- **Validation**: Smoke-tested with demonstrated finite outputs on a 10x10 grid.

### Stage B: Two-Neuron E-I Pair
- **Dynamics**: One Excitatory, one Inhibitory neuron.
- **Synapses**: AMPA/GABA conductance traces with 1.0 ms delay.
- **Validation**: Spike-triggered synaptic conductance changes verified via smoke test.

### Stage C: Population E80/I20
- **Scale**: N=100 (smoke tested with N=20).
- **Connectivity**: Sparse random connectivity.
- **Validation**: Stable firing rates and non-exploding synchrony demonstrated.

### Stage D: Cortical Column Baseline
- **Scale**: Multi-population (E, PV, SST).
- **Geometry**: Laminar position and electrode contact mapping.
- **Validation**: Stable spontaneous activity and finite LFP sampling smoke-tested.

### Stage E: SGD Adaptation
- **Method**: `optax.adam` on MSE loss between model voltage and synthetic target.
- **Constraint Mapping**: Softplus for positive params, Sigmoid for bounded.
- **Validation**: Loss decreases on synthetic target recovery.

## Limitations
- **Resistive Only**: Current solver uses frequency-independent $\sigma_e$.
- **Proxy Current**: Membrane current is a simplified proxy, not full compartmental accounting.
- **No full TFNE**: This is not full TFNE membrane current. It does not yet compute $C_m dV/dt +$ ionic currents + $I_{syn}$ with units. It is a scaffold for numerical plumbing.
- **Dense Solver**: Limited to small grids (e.g., 20x20) for smoke tests.
- **No Biological Validation Claimed**: This is a numerical scaffold, not a proven biological model.
- **Omission Application**: Blocked pending TFNE benchmarks.

## Test Results
- `manual_test_single_neuron.py`: **Smoke-tested**
- `manual_test_two_neuron.py`: **Smoke-tested**
- `manual_test_population.py`: **Smoke-tested**
- `manual_test_column.py`: **Smoke-tested**
- `manual_test_sgd.py`: **Smoke-tested**
