#!/usr/bin/env python3
"""Run homogeneous p-PQME or its ordinary-Redfield W_h=0 limit."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ppqme import PartialPolaronTransform, BathNumerics, build_model
from ppqme.core import transform_state_to_energy_basis, transform_state_from_energy_basis
from ppqme.tcl2 import cumulative_rates, build_liouvillians, propagate
from ppqme.observables import populations, minimum_eigenvalue
from models.singlet_fission import SFParameters, HAMILTONIAN_EV, HBAR_EV_FS, STATE_LABELS, independent_spectral_densities, initial_density_s1

p=argparse.ArgumentParser()
p.add_argument('--method',choices=['ppqme','redfield'],default='ppqme')
p.add_argument('--omega-h',type=float,default=0.31)
p.add_argument('--alpha',type=float,default=2.0)
p.add_argument('--temperature',type=float,default=0.0)
p.add_argument('--bath-omega-max',type=float,default=0.4)
p.add_argument('--frequency-points',type=int,default=20000)
p.add_argument('--rate-dt-fs',type=float,default=0.10)
p.add_argument('--prop-dt-fs',type=float,default=0.50)
p.add_argument('--final-time-fs',type=float,default=400.0)
p.add_argument('--output-dir',type=Path,default=None)
a=p.parse_args()

kind='partial' if a.method=='ppqme' else 'none'
out=a.output_dir or Path(f"{a.method}_T{a.temperature:g}_wmax{a.bath_omega_max:g}")
out.mkdir(parents=True,exist_ok=True)

sf=SFParameters(temperature=a.temperature)
transform=PartialPolaronTransform(omega_h=a.omega_h,alpha=a.alpha,kind=kind,frequency_points=a.frequency_points,omega_max=a.bath_omega_max)
rate_dt=a.rate_dt_fs/HBAR_EV_FS; tmax=a.final_time_fs/HBAR_EV_FS
numerics=BathNumerics(beta=sf.beta,rate_tmax=tmax,rate_dt=rate_dt)
model=build_model(HAMILTONIAN_EV,independent_spectral_densities(sf),numerics,transform)
rates_fine=cumulative_rates(model.energies,model.metadata['rate_times'],model.metadata['correlation_tensor'],model.metadata['correlation_tensor_negative'])
stride=max(1,int(round(a.prop_dt_fs/a.rate_dt_fs)))
idx=np.arange(0,len(rates_fine.times),stride)
if idx[-1] != len(rates_fine.times)-1: idx=np.append(idx,len(rates_fine.times)-1)
from ppqme.tcl2 import TimeLocalRates
rates=TimeLocalRates(rates_fine.times[idx],rates_fine.W[idx],rates_fine.W_backward[idx])
L=build_liouvillians(model,rates)
rho0e=transform_state_to_energy_basis(initial_density_s1(),model.eigenvectors)
states_e=propagate(rates.times,L,rho0e)
states_s=np.asarray([transform_state_from_energy_basis(r,model.eigenvectors) for r in states_e])
times_fs=rates.times*HBAR_EV_FS; pops=populations(states_s)
np.savetxt(out/'populations.dat',np.column_stack([times_fs,pops]),header='time_fs S1 TT CT')
print('method=',a.method,' omega_h=',a.omega_h,' T=',a.temperature,'K')
print('Debye-Waller S1-CT=',model.metadata['debye_waller'][0,2])
print(f'{a.final_time_fs:.1f} fs populations:',dict(zip(STATE_LABELS,pops[-1])))
print('minimum density eigenvalue=',minimum_eigenvalue(states_s))
fig,ax=plt.subplots(figsize=(6.5,4.4))
for i,l in enumerate(STATE_LABELS): ax.plot(times_fs,pops[:,i],label=l)
ax.set(xlabel='Time (fs)',ylabel='Population',xlim=(0,a.final_time_fs),ylim=(-0.05,1.05)); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'populations.png',dpi=300); plt.close(fig)
