from pathlib import Path
p = Path('/holdout_task/alpha_w02_batch_02_solver.py')
s = p.read_text()
a = s.index('    change = design[:, :-1] @ coef')
b = s.index("    gap = np.asarray(test['critical_strain'])", a)
s = s[:a] + '''    memory = design[:, :, :2] @ coef[:2]
    amp = design[:, -1, :2]*(taus*coef[:2])
    drift = X[n:] @ coef[2:]
    # The observed initial rate separates the coupon drift from the known
    # memory derivative when the panel cannot identify an individual feature.
    rate = np.asarray(test['instant_rate_obs'])
    incomplete = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    use_rate = incomplete & np.isfinite(rate)
    drift[use_rate] = rate[use_rate] - memory[use_rate, -1]
    pred = test0[:, None] + memory[:, :-1] + drift[:, None]*time
''' + s[b:]
p.write_text(s)
