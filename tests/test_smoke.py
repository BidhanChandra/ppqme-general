import numpy as np
from ppqme import PartialPolaronTransform,BathNumerics,build_model
from ppqme.tcl2 import cumulative_rates
from models.singlet_fission import SFParameters,HAMILTONIAN_EV,independent_spectral_densities

def test_small_model_builds():
    sf=SFParameters(temperature=0)
    tr=PartialPolaronTransform(omega_h=.31,frequency_points=500,omega_max=.4,time_chunk=32)
    num=BathNumerics(sf.beta,10.0,0.5)
    m=build_model(HAMILTONIAN_EV,independent_spectral_densities(sf),num,tr)
    assert len(m.channels)==7
    assert np.all(np.isfinite(m.metadata['phi_site']))
    r=cumulative_rates(m.energies,m.metadata['rate_times'],m.metadata['correlation_tensor'],m.metadata['correlation_tensor_negative'])
    assert np.max(np.abs(r.W[0])) < 1e-14
