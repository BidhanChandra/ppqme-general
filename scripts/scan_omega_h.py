#!/usr/bin/env python3
"""Scan homogeneous p-PQME omega_h and rank by RMSE vs digitized ML-MCTDH."""
from __future__ import annotations
import argparse,sys,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ppqme import PartialPolaronTransform,BathNumerics,build_model
from ppqme.core import transform_state_to_energy_basis, transform_state_from_energy_basis
from ppqme.tcl2 import cumulative_rates,build_liouvillians,propagate,TimeLocalRates
from ppqme.observables import populations,minimum_eigenvalue,interpolate_populations,errors
from models.singlet_fission import SFParameters,HAMILTONIAN_EV,HBAR_EV_FS,STATE_LABELS,independent_spectral_densities,initial_density_s1

p=argparse.ArgumentParser(); p.add_argument('--omega-min',type=float,default=.18); p.add_argument('--omega-max',type=float,default=.50); p.add_argument('--step',type=float,default=.01); p.add_argument('--mlmctdh',type=Path,default=ROOT/'data/mlmctdh_fig3b_digitized.csv'); p.add_argument('--output-dir',type=Path,default=Path('omega_h_scan')); a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'trajectories').mkdir(exist_ok=True)
d=np.genfromtxt(a.mlmctdh,delimiter=',',names=True); tm=np.asarray(d['time_fs']); pm=np.column_stack([d['S1'],d['TT'],d['CT']])
sf=SFParameters(temperature=0.0); vals=np.arange(a.omega_min,a.omega_max+0.5*a.step,a.step); rows=[]
for wh in vals:
    tr=PartialPolaronTransform(omega_h=float(wh),alpha=2,kind='partial',frequency_points=20000,omega_max=.4)
    num=BathNumerics(sf.beta,400/HBAR_EV_FS,.1/HBAR_EV_FS)
    m=build_model(HAMILTONIAN_EV,independent_spectral_densities(sf),num,tr)
    rf=cumulative_rates(m.energies,m.metadata['rate_times'],m.metadata['correlation_tensor'],m.metadata['correlation_tensor_negative'])
    idx=np.arange(0,len(rf.times),5); idx=np.append(idx,len(rf.times)-1) if idx[-1]!=len(rf.times)-1 else idx
    r=TimeLocalRates(rf.times[idx],rf.W[idx],rf.W_backward[idx]); L=build_liouvillians(m,r)
    se=propagate(r.times,L,transform_state_to_energy_basis(initial_density_s1(),m.eigenvectors)); ss=np.asarray([transform_state_from_energy_basis(x,m.eigenvectors) for x in se]); t=r.times*HBAR_EV_FS; pop=populations(ss)
    pred=interpolate_populations(tm,t,pop); met=errors(pm,pred)
    rows.append([wh,met['combined_rmse'],*met['rmse'],*pop[-1],minimum_eigenvalue(ss)])
    np.savetxt(a.output_dir/'trajectories'/f'ppqme_omega_h_{wh:.3f}.dat',np.column_stack([t,pop]),header='time_fs S1 TT CT')
    print(f'{wh:.3f}  RMSE={met["combined_rmse"]:.6f}')
rows=np.asarray(rows,float); rows=rows[np.argsort(rows[:,1])]
np.savetxt(a.output_dir/'omega_h_RMSE_ranking.csv',rows,delimiter=',',header='omega_h,combined_RMSE,S1_RMSE,TT_RMSE,CT_RMSE,S1_400,TT_400,CT_400,minEig',comments='')
print('\nBest:\n',rows[:10])
