"""Active learning of the higher-order correction dK = K3 + K4 across the (alpha, omega_h) frame grid.

    data       build_cache(scan) : per case Y (frames x [Re, Im] of the site-basis Schroedinger dK at the nodes),
               X = (alpha, omega_h), F = (B, lambda_p, rms||K2||), lab = (err TCL2, err TCL234, stable234, max|TCL234|)
    surrogate  PCA + one GP (Matern 5/2, ARD) per component;  acquisition strategies random / maximin / gp_u / hybrid
    study      generator-level study  (cases x strategies x seeds)          python -m goa_pqme.al.study
    pop_eval   population-level check of predicted generators             python -m goa_pqme.al.pop_eval CASE
    transfer   cross-case transfer (few-shot), optionally K2-scaled       python -m goa_pqme.al.transfer CASE ...
"""
