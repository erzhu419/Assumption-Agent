import importlib.util
from pathlib import Path

# Reuse the panel-only fitting routines, not the earlier reporting sweep.
source = Path('/tmp/mixer_audit.py').read_text().split('for entry in batch:', 1)[0]
exec(source)

root = Path('/holdout_task')
def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_node

n0, n1, n2 = [load(root / name) for name in ('_frozen_nodes/node_00.py', 'node_01.py', '_frozen_nodes/node_02.py')]
s = batch[0]['study']
train = s['train']
cal = n0({'__input__': s})
panel = cal['panel_b']
u, v = panel['u'], panel['v']
print('max abs Panel B/train/test by coordinate:', *[np.round(np.max(np.abs(a), axis=0), 3) for a in (u, cal['cal_tr'], cal['cal_te'])])
extent = np.max(np.abs(cal['cal_tr']),axis=1)
tail = extent >= np.quantile(extent,.75)
rank = np.argsort(np.max(np.abs(u),axis=1))
full = np.arange(len(u))
scenarios = {'full': [full], 'jackknife': [np.delete(full,i) for i in full], 'panel tail withheld': [rank[:-len(u)//4]]}
for mode in ('exponential',):
    for scenario, masks in scenarios.items():
        metrics = []
        for idx in masks:
            if mode == 'exponential':
                def mix(a,b):
                    return n1({'node_00': {'panel_b': {'u':u[idx], 'v':v[idx]}, 'cal_tr':a, 'cal_te':b}})
            else:
                _, _, pred = fit(u[idx],v[idx],mode)
                def mix(a,b):
                    return {'z_tr':pred(a),'z_te':pred(b)}
            for fold in range(6):
                held = np.arange(len(extent)) % 6 == fold
                study = dict(s, train={k:train[k][~held] for k in ('x','y','traj')}, test={'x':train['x'][held]})
                out = n2({'__input__':study,'node_01':mix(cal['cal_tr'][~held],cal['cal_tr'][held])})
                dy = np.abs(out['y_hat']-train['y'][held]); dt = np.abs(out['state_pred']-train['traj'][held]); t = tail[held]
                metrics.append([dy.mean(),dt.mean(),dy[t].mean(),dt[t].mean(),dt[:,-2:].mean()])
        print(mode, scenario, 'max [label, traj, tail-label, tail-traj, late-traj]',np.round(np.max(metrics,axis=0),6))
