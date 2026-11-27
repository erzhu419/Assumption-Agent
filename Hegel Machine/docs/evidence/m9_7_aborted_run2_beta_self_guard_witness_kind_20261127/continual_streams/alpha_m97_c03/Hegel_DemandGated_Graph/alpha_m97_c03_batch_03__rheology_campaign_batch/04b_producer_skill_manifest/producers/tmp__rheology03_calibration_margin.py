from pathlib import Path
from itertools import combinations
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])
cal = d['calibration_load_cells']
ref = s._prefix_calibration()
base = s.solve_alpha_m97_c03_batch_03(d)
for n_clean in (1, 2, 3, 4, 9):
    worst = (0., None)
    for lab in range(4):
        lab_rows = np.flatnonzero(cal['lab_id'] == lab)
        other = cal['lab_id'] != lab
        for rows in combinations(lab_rows, n_clean + 1):
            keep = other.copy()
            keep[list(rows)] = True
            selected = {k: v[keep] for k, v in cal.items()}
            for j in np.flatnonzero(selected['lab_id'] == lab):
                for shift in (-3.2, 3.2):
                    case = {k: v.copy() for k, v in selected.items()}
                    case['true_stress'][j] += shift
                    pred = s._calibrate(case)[lab]
                    cmd = np.r_[train['past_stress_cmd'][labs == lab].ravel(), train['stress_hold_cmd'][labs == lab]]
                    error = np.max(np.abs((pred[0]-params[lab,0])*cmd + pred[1]-params[lab,1]))
                    if error > worst[0]:
                        worst = (float(error), (lab, case, pred))
    err, (lab, case, pred) = worst
    out = s.solve_alpha_m97_c03_batch_03({**d, 'calibration_load_cells': case})
    print('clean', n_clean, 'worst lab', lab, 'stress error', round(err, 6), 'fit', pred.tolist(), 'prefix', ref[lab].tolist(), 'strain/rate max change', [round(float(np.max(np.abs(out[k]-base[k]))),6) for k in ('predicted_strain','initial_rate_pred')], 'cmd/true', [(float(a),float(b)) for a,b in zip(case['cmd_stress'][case['lab_id']==lab],case['true_stress'][case['lab_id']==lab])])
