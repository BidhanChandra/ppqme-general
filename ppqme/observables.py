"""Population/error/positivity helpers."""
from __future__ import annotations
import numpy as np

def populations(states):
    return np.real(np.diagonal(states,axis1=1,axis2=2))

def minimum_eigenvalue(states):
    return float(min(np.min(np.linalg.eigvalsh(0.5*(r+r.conj().T))) for r in states))

def interpolate_populations(target_t, source_t, source_p):
    return np.column_stack([np.interp(target_t,source_t,source_p[:,i]) for i in range(source_p.shape[1])])

def errors(reference,predicted):
    d=predicted-reference
    mae=np.mean(np.abs(d),axis=0); rmse=np.sqrt(np.mean(d*d,axis=0))
    return dict(mae=mae, rmse=rmse, combined_mae=float(np.mean(np.abs(d))), combined_rmse=float(np.sqrt(np.mean(d*d))))
