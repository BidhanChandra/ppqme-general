import numpy as np
from models.singlet_fission import HAMILTONIAN_EV, SFParameters

def test_hamiltonian():
    assert HAMILTONIAN_EV.shape == (3,3)
    assert np.allclose(HAMILTONIAN_EV,HAMILTONIAN_EV.conj().T)
    assert HAMILTONIAN_EV[0,1] == 0
    assert np.isclose(HAMILTONIAN_EV[0,2],-0.05)

def test_zero_temperature_beta():
    assert np.isinf(SFParameters(temperature=0).beta)
