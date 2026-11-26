"""Paper analyses (post-processing only; no dynamics except mechanisms --generators).

    python -m goa_pqme.analysis.build frames|coherence   scan -> figures folder (csv/*_frame_errors.csv, curves/)
    python -m goa_pqme.analysis.paper_numbers            numbers quoted in the text, paper_numbers_per_case.csv
    python -m goa_pqme.analysis.ensembles threshold|checks   robustness of the RF ensemble
    python -m goa_pqme.analysis.lg_compare               comparison with Lai & Geva
    python -m goa_pqme.analysis.positivity audit|checked|sweep
    python -m goa_pqme.analysis.frame_pattern [--figure] failure pattern of TCL234
    python -m goa_pqme.analysis.mechanisms               failure-mechanism tests
    python -m goa_pqme.analysis.ensemble_vs_single       ensemble vs single-frame rules (figure)
    python -m goa_pqme.analysis.channels                 channel-resolved (nX) analysis
    python -m goa_pqme.analysis.frame_independence       frame-ensemble spread per case
"""
