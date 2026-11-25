import sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from thermal_sink_solver import localize_thermal_source, _plate_operator, _response_bank

# Use the independent forward generator from the first verification run.
with open('/tmp/check_thermal.py') as f:
    text = f.read()
exec(text[text.index('def synthetic('):text.index('\nfor label, args in')])

def check(m, gt):
    truth = np.r_[gt['source_xy'], gt['amplitude'], gt['junction_offset']]
    fit = localize_thermal_source(m)
    estimate = np.r_[fit['source_xy'], fit['amplitude'], fit['junction_offset']]
    def error(p):
        return np.r_[np.linalg.norm(p[:2]-truth[:2]), abs(p[2]/truth[2]-1), abs(p[3]-truth[3])]
    e = error(estimate)
    if np.any(e > [.025, .05, .005]):
        bank = _response_bank(m)
        n, s = m['N'], m['sigma_m']
        c = (np.arange(n)+.5)/n
        def resid(p):
            q = np.exp(-.5*((c[:,None]-p[0])**2+(c[None,:]-p[1])**2)/s**2)
            return p[2]*np.einsum('mij,ij->m',bank,q)+p[3]-m['T_obs']
        good = np.abs(resid(truth)) < .02
        oracle = least_squares(lambda p: resid(p)[good], truth, bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf]), x_scale='jac')
        dist = np.linalg.norm(c[m['sensor_ij']]-truth[:2],axis=1)
        ri = resid(estimate)
        print(label, seed, 'solver', np.round(e,6), 'oracle', np.round(error(oracle.x),6), 'closest clean distances', np.round(np.sort(dist[good])[:3],4), 'largest clean residual', round(np.max(abs(ri[good])),6), flush=True)
    return e

for label, args, seeds in [
    ('original corner', (35,.09,7.,(.96,.045),64,1.1,.08), range(6)),
    ('strong Robin corner', (35,.09,7.,(.96,.045),64,1.1,.06), range(6,36)),
    ('opposite corner', (27,.085,7.,(.055,.94),52,1.1,.06), range(20)),
    ('weak Robin edge', (29,.085,.45,(.043,.72),48,1.1,.06), range(6,22)),
    ('narrow sparse near corner', (31,.055,2.2,(.09,.91),40,.8,.06), range(16)),
]:
    rows = []
    for seed in seeds:
        rows.append(synthetic(*args, seed))
    rows = np.array(rows)
    print(label, 'worst xy/rel_amp/offset',rows.max(axis=0), 'misses',int(np.any(rows > [.035,.08,.008],axis=1).sum()),'of',len(rows), flush=True)
