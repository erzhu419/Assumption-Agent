import itertools
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _Rosette, _cyclic_design, _robust_fit
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
pairs = list(itertools.combinations_with_replacement(range(4), 2))
def designs(x, family, q=None, strength=0.):
    base = [_cyclic_design(x, j) for j in range(4)]
    if family == 'cyclic cubic':
        return base
    if family == 'all quadratic contrasts':
        # All quadratic monomials span the squared pairwise sums/differences.
        quad = np.column_stack([x[:, i]*x[:, j] for i,j in pairs])
        return [np.column_stack((a[:, :4], quad, a[:, -1])) for a in base]
    return [np.column_stack((a[:, :4], a[:, 4:] / (1 + strength*(x @ q[j])**2)[:, None])) for j,a in enumerate(base)]
eye = np.eye(4)
configs = [(np.roll(eye, -1, axis=0), t) for t in (.05, .2, .8)]
configs += [(eye, t) for t in (.05, .2, .8)]
configs += [(eye + sign*np.roll(eye, -1, axis=0), t) for sign in (-1, 1) for t in (.05, .2, .8)]
def prediction(x, y, tr, te, family, config=()):
    a = designs(x, family, *config)
    c, _ = _robust_fit([z[tr] for z in a], y[tr])
    return np.column_stack([z[te] @ c[:, j] for j,z in enumerate(a)])
def rms(r):
    return np.sqrt(np.mean(r*r, axis=0)).max()
for item in items:
    study = item['study']
    s = study['sensor_calibration_sweeps']; x, y = s['ref_rosette'], s['raw_rosette']
    n = np.arange(len(x)); axis = np.count_nonzero(x, axis=1) == 1
    # Published fixture's gross spikes are separated from valid readings by amplitude.
    clean = np.max(np.abs(y), axis=1) < 7
    families = ('cyclic cubic', 'all quadratic contrasts', 'rational saturation')
    worst = {name: 0. for name in families}; inv_worst = 0.
    for held in [n%4 == k for k in range(4)] + [axis]:
        tr = ~held
        for family in families:
            config = ()
            if family == 'rational saturation':
                def inner_score(cfg):
                    scores = []
                    for k in range(3):
                        val = tr & (n%3 == k)
                        pred = prediction(x,y,tr & ~val,val,family,cfg)
                        # Use a robust score for selecting saturation without using outer labels.
                        scores.append(np.median(np.linalg.norm(pred-y[val], axis=1)))
                    return max(scores)
                config = min(configs, key=inner_score)
            for slice_mask in (axis, ~axis):
                te = held & clean & slice_mask
                if np.any(te):
                    err = rms(prediction(x,y,tr,te,family,config)-y[te])
                    worst[family] = max(worst[family],err)
                    if family == 'cyclic cubic':
                        inv_worst = max(inv_worst, rms(_Rosette(x[tr],y[tr]).invert(y[te])-x[te]))
    fit = _Rosette(x,y)
    raw_blocks = [study['y_obs_raw']] + [p['y_pulse_raw'] for p in study.get('reference_tracer_pulses',[]) if p['y_pulse_raw'] is not None]
    ref_blocks = [fit.invert(b).reshape(-1,4) for b in raw_blocks]
    sigma_min = min(np.linalg.svd(fit.jacobian(b),compute_uv=False)[:,-1].min() for b in ref_blocks)
    print('Worst held-out forward channel RMSE by family:',worst)
    print('Worst cyclic inverse channel RMSE:',inv_worst,'; smallest Jacobian singular value on measurement and pulse blocks:',sigma_min)
    print('Study fields:', ', '.join(sorted(study)))
    assert inv_worst < 1e-4
