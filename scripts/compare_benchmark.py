#!/usr/bin/env python3
"""Compare p-PQME omega_h=0.31, Redfield, and digitized ML-MCTDH."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ppqme.observables import interpolate_populations,errors
pp=np.loadtxt('omega_h_scan/trajectories/ppqme_omega_h_0.310.dat'); rd=np.loadtxt('redfield_T0_wmax0.4/populations.dat')
d=np.genfromtxt(ROOT/'data/mlmctdh_fig3b_digitized.csv',delimiter=',',names=True); tm=d['time_fs']; pm=np.column_stack([d['S1'],d['TT'],d['CT']])
for name,x in [('p-PQME',pp),('Redfield',rd)]:
    met=errors(pm,interpolate_populations(tm,x[:,0],x[:,1:4])); print(name,met)
fig,ax=plt.subplots(figsize=(7.2,4.8)); labels=['S1','TT','CT']
for i,l in enumerate(labels):
    line,=ax.plot(pp[:,0],pp[:,i+1],lw=2,label=f'p-PQME {l}')
    ax.plot(rd[:,0],rd[:,i+1],'--',color=line.get_color(),label=f'Redfield {l}')
    ax.plot(tm,pm[:,i],'o',mfc='none',color=line.get_color(),label=f'ML-MCTDH {l}')
ax.set(xlabel='Time (fs)',ylabel='Population',xlim=(0,400),ylim=(-.05,1.05)); ax.legend(frameon=False,ncol=3,fontsize=8); fig.tight_layout(); fig.savefig('comparison.png',dpi=350)
