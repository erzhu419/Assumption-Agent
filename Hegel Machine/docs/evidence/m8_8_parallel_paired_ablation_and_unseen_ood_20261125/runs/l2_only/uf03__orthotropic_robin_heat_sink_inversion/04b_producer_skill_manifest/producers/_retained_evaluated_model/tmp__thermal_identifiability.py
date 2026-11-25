import sys
sys.path.insert(0, '/holdout_task')
sys.path.append('/tmp')
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from thermal_sink_solver import localize_thermal_source, _plate_operator, _response_bank
from thermal_wider_gate import localize_thermal_source as wider
with open('/tmp/check_thermal.py') as f:
    text = f.read()
exec(text[text.index('def synthetic('):text.index('\nfor label, args in')])

def check(m, gt):
    truth = np.r_[gt['source_xy'], gt['amplitude'], gt['junction_offset']]
    def err(fit):
        return [np.linalg.norm(np.subtract(fit['source_xy'], truth[:2])), abs(fit['amplitude']/truth[2]-1), abs(fit['junction_offset']-truth[3])]
    e = np.array([err(s(m)) for s in (localize_thermal_source, wider)])
    if np.any(e > [.035, .08, .008]):
        bank = _response_bank(m)
        c = (np.arange(m['N'])+.5)/m['N']; s = m['sigma_m']
        def predict(x, z, a, b):
            q = np.exp(-0.5*((c[:, None]-x)**2+(c[None, :]-z)**2)/s**2)
            return a*np.einsum('mij,ij->m', bank, q)+b
        mean = predict(*truth)
        good = abs(m['T_obs']-mean) < .02
        alternatives = []
        for ratio in (.84, 1.18):
            def diff(p):
                return (predict(p[0], p[1], truth[2]*ratio, p[2])-mean)[good]
            fit = least_squares(diff, truth[[0,1,3]], bounds=([0,0,-np.inf],[1,1,np.inf]), gtol=1e-12)
            alternatives.append(np.linalg.norm(fit.fun)/.001)
        print(label, seed, 'baseline / wider [xy, amp, offset]', np.round(e,6).tolist(), 'clean mean distance / noise at 0.84x,1.18x amplitude', np.round(alternatives,3).tolist())
    return e

for label, args, seeds in [
    ('corner', (35,.09,7.,(.96,.045),64,1.1,.06), range(36)),
    ('opposite corner', (27,.085,7.,(.055,.94),52,1.1,.06), range(20)),
    ('narrow near corner', (31,.055,2.2,(.09,.91),40,.8,.06), range(16)),
    ('public-like interior', (22,.085,2.2,(.37,.65),52,.8,.06), range(12)),
]:
    rows = []
    for seed in seeds:
        rows.append(synthetic(*args, seed))
    a = np.array(rows)
    print(label, 'worst baseline / wider', a.max(axis=0).tolist(), 'misses', np.any(a > [.035,.08,.008],axis=2).sum(axis=0).tolist(), 'of', len(rows), flush=True)
