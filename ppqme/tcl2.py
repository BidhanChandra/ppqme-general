"""Finite-time homogeneous TCL2 p-PQME/Redfield propagation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from .core import PTModel, operator_superoperator, energy_hamiltonian_superoperator

@dataclass(frozen=True)
class TimeLocalRates:
    times: np.ndarray
    W: np.ndarray
    W_backward: np.ndarray


def _cumtrapz(y,t):
    dt=np.diff(t); out=np.zeros_like(y,complex)
    out[...,1:] = np.cumsum(0.5*(y[...,1:]+y[...,:-1])*dt, axis=-1)
    return out


def cumulative_rates(energies, times, corr, corr_neg):
    E=np.asarray(energies,float); t=np.asarray(times,float)
    delta=E[:,None]-E[None,:]
    phase=np.exp(-1j*delta[:,:,None]*t[None,None,:])
    f = corr[:,:,None,None,:]*phase[None,None,:,:,:]
    b = corr_neg[:,:,None,None,:]*phase[None,None,:,:,:]
    return TimeLocalRates(t, np.moveaxis(_cumtrapz(f,t),-1,0), np.moveaxis(_cumtrapz(b,t),-1,0))


def dissipator_at_rates(model: PTModel, W, Wb):
    S=[c.system_operator for c in model.channels]; nc=len(S); n=len(model.energies)
    A=[[W[a,b]*S[b] for b in range(nc)] for a in range(nc)]
    B=[[Wb[b,a]*S[b] for b in range(nc)] for a in range(nc)]
    def action(rho):
        out=np.zeros_like(rho,complex)
        for a,Sa in enumerate(S):
            for b in range(nc):
                out += A[a][b]@rho@Sa - Sa@A[a][b]@rho - rho@B[a][b]@Sa + Sa@rho@B[a][b]
        return out
    return operator_superoperator(n,action)


def build_liouvillians(model: PTModel, rates: TimeLocalRates):
    U0=energy_hamiltonian_superoperator(model.energies)
    L=np.empty((len(rates.times),len(model.energies)**2,len(model.energies)**2),complex)
    for i in range(len(rates.times)):
        L[i]=U0+dissipator_at_rates(model,rates.W[i],rates.W_backward[i])
    return L


def propagate(grid_times, L, rho0):
    t=np.asarray(grid_times,float); rho=np.asarray(rho0,complex); n=rho.shape[0]
    v=rho.reshape(-1,order='F'); out=np.empty((len(t),n,n),complex); out[0]=rho
    for i in range(len(t)-1):
        dt=t[i+1]-t[i]; Lmid=0.5*(L[i]+L[i+1])
        v=expm(Lmid*dt)@v; out[i+1]=v.reshape((n,n),order='F')
    return out
