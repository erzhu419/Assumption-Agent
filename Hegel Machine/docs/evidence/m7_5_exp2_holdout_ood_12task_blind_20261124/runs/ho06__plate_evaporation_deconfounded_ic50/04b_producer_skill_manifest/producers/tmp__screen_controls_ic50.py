from pathlib import Path
import numpy as np
import pickle

# Reuse the same simulated plates and oracle fitting setup for paired comparisons.
setup = Path('/tmp/diagnose_ic50.py').read_text().split('for name, estimator in ')[0]
exec(compile(setup, '/tmp/diagnose_ic50.py', 'exec'))
with open('/holdout_task/public_screen.pkl', 'rb') as f:
    public = pickle.load(f)
needle = '        signal = fluorescence - background\n        reference = np.median(signal[positive])'
assert needle in source

def candidate(penalty, cutoff, ctrl_scale):
    code = source.replace(needle, '        signal = fluorescence - background\n        positive &= signal > ' + repr(cutoff) + ' * np.median(signal[positive])\n        reference = np.median(signal[positive])')
    code = code.replace("jac=lambda coef: design, loss='cauchy', f_scale=0.025)", "jac=lambda coef: design, loss='cauchy', f_scale=" + repr(ctrl_scale) + ')')
    if penalty is not None:
        code = code.replace(shared, shared + '\n    common_decay = decay')
        code = code.replace(loop, '''    for wells, row, col, kind, signal, dmso, reference in prepared:
        def local_profile(t):
            return (_fit_control_gain(row[dmso], col[dmso], signal[dmso], t).cost
                    + PENALTY * np.log(t / common_decay)**2)
        decay = minimize_scalar(local_profile, bounds=(0.5, 8.), method='bounded',
                                options={'xatol': 0.005}).x
        coef ='''.replace('PENALTY', repr(penalty)))
    scope = {}
    exec(compile(code, '<control-screening candidate>', 'exec'), scope)
    return scope['estimate_compound_hill_parameters']

def public_error(est):
    out = est(public['plates'])
    keys = ('log10_ic50_uM','hill_slope')
    return np.max([[abs(out[k][field] - v[field]) for field in keys] for k,v in public['gt'].items()],axis=0)

for penalty, cutoff, ctrl_scale in [(None,.50,.025),(0.,.50,.025),(.0005,.50,.025),(.003,.50,.025),(0.,.50,.015),(.0005,.50,.015)]:
    est = candidate(penalty,cutoff,ctrl_scale)
    namespace['estimate'] = est
    print('candidate',penalty,cutoff,ctrl_scale,'public',np.round(public_error(est),5),flush=True)
    for varied in (False,True):
        errors=np.array([namespace['simulate'](seed,varied) for seed in range(30)])
        print('varied',varied,'max',np.round(errors.max((0,1)),5),'failed seed/compound',np.argwhere(np.any(errors>[.09,.18],axis=-1)).tolist(),flush=True)
