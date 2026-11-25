import numpy as np
from collections import defaultdict

path = '/tmp/check_tensor_darcy.py'
text = open(path).read()
marker = 'worst_balance = 0.'
head, tail = text.split(marker, 1)
exec(head)
original_errors = errors
comparisons = defaultdict(list)

def errors(p, weights, offset, field):
    baseline = original_errors(p, weights, offset, field)
    ij = tuple(p['sensor_ij'].T)
    y = p['y_obs']
    true_residual = y - field[ij] - offset
    clean = abs(true_residual) < .75
    b = mod.infer_tensor_darcy_sources(p)
    detected = abs(y - b['predicted_field'][ij] - b['sensor_offset']) < .75
    oracle = original_errors(dict(p, sensor_ij=p['sensor_ij'][clean], y_obs=y[clean]), weights, offset, field)
    responses = mod._forward(p)
    flat = np.ravel_multi_index(ij, field.shape)
    a = np.column_stack((responses[flat,1:], np.ones(len(flat))))
    z = y - responses[flat,0]
    pmat = np.diag(np.r_[np.full(len(weights), np.sqrt(p['tikhonov_alpha'])), 0.])
    coef = np.linalg.lstsq(np.vstack((a[detected], pmat)), np.r_[z[detected], np.zeros(len(pmat))], rcond=None)[0]
    pred = (responses[:,0] + responses[:,1:] @ coef[:-1]).reshape(field.shape)
    gapfit = (np.linalg.norm(coef[:-1]-weights)/np.linalg.norm(weights), abs(coef[-1]-offset), np.linalg.norm(pred-field)/np.linalg.norm(field))
    comparisons[(p['Lx'], p['gamma_robin'], np.sign(p['k_xy'].mean()))].append((baseline, oracle, gapfit, np.mean(b['predicted_field'] - field), np.sqrt(np.mean(field**2)), np.sum(clean != detected)))
    return baseline

exec(marker + tail)
for key, rows in comparisons.items():
    vals = np.asarray([r[:3] for r in rows]); w = max(rows, key=lambda r:r[0][2]); print('paired slice',key,'worst source/offset/field by baseline, oracle mask with current fit, detected mask with quadratic refit',vals.max(axis=0).round(7).tolist(), 'worst field mean error and truth RMS', w[3:5], 'mask disagreements', sum(r[5] for r in rows))
