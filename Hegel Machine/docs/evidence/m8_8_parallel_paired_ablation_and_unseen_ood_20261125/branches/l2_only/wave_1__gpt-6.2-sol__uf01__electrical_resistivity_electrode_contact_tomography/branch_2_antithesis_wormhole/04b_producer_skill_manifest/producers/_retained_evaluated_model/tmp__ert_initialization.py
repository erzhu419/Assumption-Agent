from pathlib import Path
p = Path('/holdout_task/ert_solver.py').read_text()
p = p.replace("return value - self.q @ (self.q.T @ value)", "return value - np.einsum('ij,j->i', self.q, np.einsum('ij,i->j', self.q, value))")
p = p.replace('    keep = np.abs(voltage - design @ coef) < threshold', '''    fitted = design @ coef
    # Give large residuals bounded influence before making a binary mask.
    for _ in range(12):
        w = np.sqrt(np.minimum(1., .011 / np.maximum(np.abs(voltage-fitted), 1e-15)))
        coef = np.linalg.lstsq(design*w[:, None], voltage*w, rcond=None)[0]
        updated = design @ coef
        if np.max(np.abs(updated-fitted)) < 1.e-6:
            fitted = updated
            break
        fitted = updated
    keep = np.abs(voltage - fitted) < threshold''')
Path('/tmp/ert_robust_init.py').write_text(p)
s = Path('/tmp/ert_stress.py').read_text()
s = s.replace('from ert_recovered import invert_ert_anomaly as recovered','from ert_robust_init import invert_ert_anomaly as recovered').replace("('recover', recovered)","('robust init', recovered)")
s = s.replace("'strong anomaly'):","'strong anomaly', 'wall plus strong', 'clustered spikes'):")
s = s.replace("if regime == 'near walls':", "if regime in ('near walls', 'wall plus strong'):")
s = s.replace("if regime == 'strong anomaly': amp = rng.uniform(12.,22.)", "if regime in ('strong anomaly', 'wall plus strong'): amp = rng.uniform(12.,22.)")
s = s.replace('        step = .06', '''        signs = rng.choice([-1,1],len(spikes))
        if regime == 'clustered spikes':
            col = rng.integers(n)
            count = 4 + rep % 3
            cluster = rng.choice(m,count,replace=False)*n + col
            other = np.flatnonzero(np.arange(m*n)%n != col)
            spikes = np.r_[cluster,rng.choice(other,len(spikes)-count,replace=False)]
            signs[:count] = signs[0]
        step = .06''')
s = s.replace('v.flat[spikes] += rng.choice([-1,1],len(spikes))*step','v.flat[spikes] += signs*step')
s = s.replace('        signs = rng.choice([-1,1],len(spikes))','        if regime != "variable spikes": signs = rng.choice([-1,1],len(spikes))')
s = s.replace('        v.flat[spikes] += signs*step','        if regime == "variable spikes": signs = rng.choice([-1,1],len(spikes))\n        v.flat[spikes] += signs*step')
Path('/tmp/ert_init_stress.py').write_text(s)
