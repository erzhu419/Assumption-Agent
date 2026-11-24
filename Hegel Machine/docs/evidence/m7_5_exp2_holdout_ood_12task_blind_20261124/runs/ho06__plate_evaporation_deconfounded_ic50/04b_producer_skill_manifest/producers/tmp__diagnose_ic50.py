import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, '/holdout_task')
import assay_ic50 as base

source = Path(base.__file__).read_text()
shared = '''    decay = minimize_scalar(profile, bounds=(0.5, 8.), method='bounded',
                            options={'xatol': 0.005}).x'''
loop = '    for wells, row, col, kind, signal, dmso, reference in prepared:\n        coef ='
assert shared in source and loop in source

def adaptive(penalty):
    code = source.replace(shared, shared + '\n    common_decay = decay')
    code = code.replace(loop, '''    for wells, row, col, kind, signal, dmso, reference in prepared:
        def local_profile(t):
            cost = _fit_control_gain(row[dmso], col[dmso], signal[dmso], t).cost
            return cost + PENALTY * np.log(t / common_decay)**2
        decay = minimize_scalar(local_profile, bounds=(0.5, 8.), method='bounded',
                                options={'xatol': 0.005}).x
        coef ='''.replace('PENALTY', repr(penalty)))
    namespace = {}
    exec(compile(code, '<adaptive candidate>', 'exec'), namespace)
    return namespace['estimate_compound_hill_parameters']

stress = Path('/tmp/stress_ic50.py').read_text().split('for varied in [False,True]:')[0]
stress = stress.replace('    plates=[]', '    plates=[]\n    oracle={}').replace("            wells.append(dict", "            if kind == 'COMPOUND': oracle.setdefault(compound, []).append((dose, (y-bg)/gain[i], gain[i]))\n            wells.append(dict")
stress = stress.replace('    pred=estimate(plates)', '''    if estimate is None:
        pred={}
        for k,a in oracle.items():
            p,s=base._fit_hill(*np.array(a).T)
            pred[k]={'log10_ic50_uM':p,'hill_slope':s}
    else:
        pred=estimate(plates)''')
stress = stress.replace('    return error.max(axis=0)', '    return error')
namespace = {'base': base}
exec(compile(stress, '<diagnostic stress>', 'exec'), namespace)
for name, estimator in [('shared',base.estimate_compound_hill_parameters),('local',adaptive(0.0)),('weakly pooled',adaptive(.0005)),('partly pooled',adaptive(.003)),('oracle gain',None)]:
    namespace['estimate']=estimator
    for varied in (False,True):
        errors=np.array([namespace['simulate'](seed,varied) for seed in range(30)])
        limits=np.array([.09,.18])
        print(name, 'varied',varied,'max',np.round(errors.max((0,1)),5), 'failed seed/compound', np.argwhere(np.any(errors>limits,axis=-1)).tolist(),flush=True)
