import numpy as np
from ppqme import DebyeSpectralDensity, localized_initial_state, run_simulation

def test_arbitrary_two_state_smoke():
    H = np.array([[0.0, 0.02], [0.02, 0.10]])
    baths = [DebyeSpectralDensity(0.01, 0.18) for _ in range(2)]
    result = run_simulation(
        H, baths, localized_initial_state(2, 0),
        temperature_kelvin=300,
        method="ppqme",
        omega_h=0.1,
        bath_omega_max=0.5,
        frequency_points=400,
        rate_dt_fs=0.5,
        prop_dt_fs=1.0,
        final_time_fs=2.0,
    )
    assert result.populations.shape == (3, 2)
    assert result.states_site.shape == (3, 2, 2)
    assert result.trace_error < 1e-10

def test_site_specific_baths():
    H = np.array([[0.0, 0.01], [0.01, 0.05]])
    baths = [DebyeSpectralDensity(0.01, 0.12), DebyeSpectralDensity(0.03, 0.25)]
    result = run_simulation(
        H, baths, localized_initial_state(2, 1),
        final_time_fs=1.0, rate_dt_fs=0.5, prop_dt_fs=0.5,
        frequency_points=400, bath_omega_max=0.5,
    )
    assert len(result.model.metadata["lambda_partial"]) == 2
