"""Zheng et al. three-state singlet-fission benchmark model.

State order: S1, TT, CT.
All energies are eV.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

HBAR_EV_FS = 0.6582119569
KB_EV_K = 8.617333262145e-5
STATE_LABELS = ('S1','TT','CT')

HAMILTONIAN_EV = np.array([
    [ 0.20,  0.00, -0.05],
    [ 0.00,  0.00, -0.05],
    [-0.05, -0.05,  0.30],
], dtype=complex)

@dataclass(frozen=True)
class SFParameters:
    reorganization_energy: float = 0.10
    omega_c: float = 0.18
    temperature: float = 0.0

    @property
    def beta(self):
        if self.temperature < 0:
            raise ValueError('temperature must be >=0')
        return np.inf if self.temperature == 0 else 1/(KB_EV_K*self.temperature)

    def spectral_density(self, omega):
        w=np.asarray(omega,float)
        return 2*self.reorganization_energy*w*self.omega_c/(w*w+self.omega_c*self.omega_c)


def independent_spectral_densities(params: SFParameters):
    return [params.spectral_density]*3


def initial_density_s1():
    rho=np.zeros((3,3),complex); rho[0,0]=1.0; return rho
