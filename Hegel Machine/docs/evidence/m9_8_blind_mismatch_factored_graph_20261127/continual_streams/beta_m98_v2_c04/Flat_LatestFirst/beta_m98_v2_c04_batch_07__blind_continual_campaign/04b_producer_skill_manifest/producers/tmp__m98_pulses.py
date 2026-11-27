import importlib.util
import pathlib
import pickle
import numpy as np
p = pathlib.Path('/tmp/hegel_continual_archive')
mods = {}
for f in sorted(p.glob('*_solver.py')):
    name = f.stem.removeprefix('beta_m98_v2_c04_').removesuffix('_solver')
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mods[name] = mod
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
pulses = item['campaign_diagnostics']
Z = s['Z_nuisance']
Q, _ = np.linalg.qr(Z, mode='reduced')
S = {k:m._forward(s)[0] for k,m in mods.items()}
for a,mod in mods.items():
    try:
        y = [mod._calibrate(dict(s, y_obs_raw=t['y_pulse_raw'])) for t in pulses]
    except Exception as exc:
        print(a, repr(exc)); continue
    for b,M in S.items():
        errs = [v-M@t['w_pulse'] for v,t in zip(y,pulses)]
        rm = lambda r: np.sqrt(np.mean(r*r))
        print(f'{a:8s} {b:8s} RMSE '+','.join(f'{rm(e):.7g}' for e in errs)+' nuisance-free '+','.join(f'{rm(e-Q@(Q.T@e)):.7g}' for e in errs))
