# TFNE Numerical Smoke Receipt V0

## A. Pure Neumann Poisson
- **claim**: NUMERICAL_SMOKE_VALIDATED_FOR_PURE_NEUMANN_POISSON_ONLY
- **residual**: 1.95e-05
- **ridge**: absent
- **RHS projection**: b - mean(b)
- **gauge**: mean-zero phi
- **nonclaim**: not TFNE-wide validation

## B. Passive cable 1D
- **claim**: NUMERICAL_SMOKE_VALIDATED_FOR_PASSIVE_CABLE_1D_ONLY
- **n=50**: relative L2 0.00011582 max abs 0.0001561
- **n=100**: relative L2 0.00007822261 max abs 0.0001713
- **n=200**: relative L2 0.00009380292 max abs 0.00027549
- **note**: nonmonotonic finest-grid saturation
- **nonclaim**: not biological validation

## C. Passive RC membrane impedance
- **claim**: NUMERICAL_SMOKE_VALIDATED_FOR_PASSIVE_RC_MEMBRANE_IMPEDANCE_ONLY
- **1 Hz**: rel amp 0.0000 phase 0.0000
- **5 Hz**: rel amp 0.0004 phase 0.0001
- **10 Hz**: rel amp 0.0014 phase 0.0009
- **50 Hz**: rel amp 0.0046 phase 0.0143
- **100 Hz**: rel amp 0.0051 phase 0.0307
- **nonclaim**: passive RC only, not HH gated impedance

## D. Anisotropic resistive field steering
- **claim**: NUMERICAL_SMOKE_VALIDATED_FOR_ANISOTROPIC_RESISTIVE_FIELD_STEERING_ONLY
- **isotropic ratio**: 1.0000
- **x-steer ratio**: 0.7409
- **y-steer ratio**: 1.3498
- **source**: centered Gaussian with mean-zero projection
- **nonclaim**: field steering only, not biological validation

*Note: None of these benchmarks constitute TFNE-wide validation, biological validation, HH gated impedance, active cable validation, or omission readiness. No empirical fitting performed.*
