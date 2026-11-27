exec(open('/tmp/check_alpha_batch.py').read().split("run('public', d)")[0])
# Withhold labels, not reference coupons: the campaign supplies the whole panel.
for j in range(4):
    case = copy.deepcopy(d)
    tr = case['train']
    mask = tr['lab_id'] == j
    for key in ('strain_curves_obs', 'instant_rate_obs'):
        tr[key] = np.asarray(tr[key], dtype=float)
        tr[key][mask] = np.nan
    run('withhold labels lab '+str(j), case)
# Stress both wings with one-check calibration, preserving physical stresses.
case = copy.deepcopy(d)
c = case['calibration_load_cells']
a = np.array([1.07, .93, 1.09, .91]); b = np.array([.08, -.09, .14, -.12])
for split in (case['train'], case['test'], c):
    ids = split['lab_id']
    for key in ('past_stress_cmd', 'stress_hold_cmd', 'cmd_stress'):
        if key in split:
            x = np.asarray(split[key]); shape = (len(ids),) + (1,)*(x.ndim-1)
            split[key] = (x-b[ids].reshape(shape))/a[ids].reshape(shape)
for sparse in ((0, 2), (1, 3)):
    keep = ~np.isin(c['lab_id'], sparse)
    for j in sparse:
        ix = np.flatnonzero(c['lab_id'] == j)
        keep[ix[np.argmax(abs(c['cmd_stress'][ix]))]] = True
    run('one check labs '+str(sparse), dict(case, calibration_load_cells={k:v[keep] for k,v in c.items()}))
