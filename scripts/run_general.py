#!/usr/bin/env python3
"""Run an arbitrary N-state p-PQME model from a JSON configuration."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppqme.config import load_config, build_from_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
parser.add_argument("--output-dir", type=Path, default=Path("results"))
args = parser.parse_args()

cfg = load_config(args.config)
labels, result = build_from_config(cfg)
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)

np.savetxt(
    out / "populations.dat",
    np.column_stack([result.times_fs, result.populations]),
    header="time_fs " + " ".join(labels),
)

np.save(out / "density_matrices.npy", result.states_site)

with open(out / "diagnostics.txt", "w", encoding="utf-8") as f:
    f.write(f"trace_error = {result.trace_error:.12e}\n")
    f.write(f"hermiticity_error = {result.hermiticity_error:.12e}\n")
    f.write(f"minimum_density_eigenvalue = {result.minimum_density_eigenvalue:.12e}\n")
    f.write("final_populations:\n")
    for label, value in zip(labels, result.populations[-1]):
        f.write(f"  {label}: {value:.12g}\n")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for i, label in enumerate(labels):
    ax.plot(result.times_fs, result.populations[:, i], label=label)
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Population")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "populations.png", dpi=300)
plt.close(fig)

print("states =", len(labels))
print("final populations =", dict(zip(labels, result.populations[-1])))
print("trace error =", result.trace_error)
print("minimum density eigenvalue =", result.minimum_density_eigenvalue)
print("results written to", out)
