#!/usr/bin/env python3
"""Validate I1 and I2 for the localized |S1><S1| initial condition.

This is intentionally separate from the production scan because I2 is expensive.
Default: T=0 K, omega_max=0.4 eV, omega_h=0.31 eV, 400 fs.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ppqme import PartialPolaronTransform,BathNumerics,build_model
from ppqme.core import transform_state_to_energy_basis,transform_state_from_energy_basis
from ppqme.tcl2 import cumulative_rates,build_liouvillians,TimeLocalRates,propagate
from ppqme.observables import populations,minimum_eigenvalue
from advanced.inhomogeneous_first_order import first_order_inhomogeneous_source,downsample_first_order_source,propagate_time_local_affine_piecewise_midpoint,first_order_source_diagnostics
from advanced.inhomogeneous_second_order import second_order_inhomogeneous_source_local_initial,second_order_source_diagnostics
from models.singlet_fission import SFParameters,HAMILTONIAN_EV,HBAR_EV_FS,independent_spectral_densities,initial_density_s1,STATE_LABELS

ap=argparse.ArgumentParser(); ap.add_argument('--omega-h',type=float,default=.31); ap.add_argument('--final-time-fs',type=float,default=400.); ap.add_argument('--i2-dt-fs',type=float,default=.5); a=ap.parse_args()
sf=SFParameters(temperature=0); tr=PartialPolaronTransform(omega_h=a.omega_h,alpha=2,kind='partial',frequency_points=20000,omega_max=.4); num=BathNumerics(sf.beta,a.final_time_fs/HBAR_EV_FS,.1/HBAR_EV_FS)
m=build_model(HAMILTONIAN_EV,independent_spectral_densities(sf),num,tr)
rf=cumulative_rates(m.energies,m.metadata['rate_times'],m.metadata['correlation_tensor'],m.metadata['correlation_tensor_negative'])
stride=max(1,int(round(a.i2_dt_fs/.1))); idx=np.arange(0,len(rf.times),stride); idx=np.append(idx,len(rf.times)-1) if idx[-1]!=len(rf.times)-1 else idx
rates=TimeLocalRates(rf.times[idx],rf.W[idx],rf.W_backward[idx]); L=build_liouvillians(m,rates); rho0s=initial_density_s1(); rho0e=transform_state_to_energy_basis(rho0s,m.eigenvectors)
hom=propagate(rates.times,L,rho0e)
i1full=first_order_inhomogeneous_source(m,rho0s); i1=downsample_first_order_source(i1full,idx)
i2=second_order_inhomogeneous_source_local_initial(m,idx,initial_site=0,progress=True)
plus1=propagate_time_local_affine_piecewise_midpoint(rates.times,L,i1.source_energy,rho0e)
full=propagate_time_local_affine_piecewise_midpoint(rates.times,L,i1.source_energy+i2.source_energy,rho0e)
def site(states): return np.asarray([transform_state_from_energy_basis(r,m.eigenvectors) for r in states])
for name,states in [('hom',hom),('+I1',plus1),('+I1+I2',full)]:
    ps=populations(site(states)); print(name,dict(zip(STATE_LABELS,ps[-1])),'minEig=',minimum_eigenvalue(site(states)))
print(first_order_source_diagnostics(i1)); print(second_order_source_diagnostics(i2))
