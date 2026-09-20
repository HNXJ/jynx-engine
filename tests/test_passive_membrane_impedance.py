import jax.numpy as jnp
from tfne_jax.validation.passive_membrane_impedance import passive_membrane_impedance, estimate_impedance

def test_multi_freq_impedance():
    frequencies = [1.0, 5.0, 10.0, 50.0, 100.0]
    Cm = 1.0e-6
    gL = 1.0e-4
    
    print(f"{'Freq':<10} | {'NumAmp':<10} | {'AnaAmp':<10} | {'RelAmpErr':<10} | {'NumPh':<10} | {'AnaPh':<10} | {'AbsPhErr':<10}")
    
    for f in frequencies:
        ana_z = passive_membrane_impedance(f, Cm, gL)
        ana_amp = jnp.abs(ana_z)
        ana_ph = jnp.angle(ana_z)
        
        num_amp, num_ph = estimate_impedance(f, Cm, gL)
        
        rel_amp_err = jnp.abs(num_amp - ana_amp) / ana_amp
        abs_ph_err = jnp.abs(num_ph - ana_ph)
        
        print(f"{f:<10} | {num_amp:.4e} | {ana_amp:.4e} | {rel_amp_err:.4f} | {num_ph:.4f} | {ana_ph:.4f} | {abs_ph_err:.4f}")
        
        assert rel_amp_err < 0.05, f"Amp error at {f}Hz: {rel_amp_err}"
        assert abs_ph_err < 0.1, f"Phase error at {f}Hz: {abs_ph_err}"
        assert jnp.isfinite(num_amp) and jnp.isfinite(num_ph)
    
    print("PASS")

if __name__ == "__main__":
    test_multi_freq_impedance()
