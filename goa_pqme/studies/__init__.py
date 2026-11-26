"""Numerical studies built on goa_pqme.run_frame / run_tempo.

    python -m goa_pqme.studies.ladder      convergence ladder (node spacing, nq, panel width)
    python -m goa_pqme.studies.bias        bias scan (TEMPO + channel-resolved TCL at lab/mid/dressed frames), lab-frame control
    python -m goa_pqme.studies.tempo_conv  TEMPO time-step / truncation convergence
    python -m goa_pqme.studies.checks      trace and Hermiticity of the propagated state
"""
