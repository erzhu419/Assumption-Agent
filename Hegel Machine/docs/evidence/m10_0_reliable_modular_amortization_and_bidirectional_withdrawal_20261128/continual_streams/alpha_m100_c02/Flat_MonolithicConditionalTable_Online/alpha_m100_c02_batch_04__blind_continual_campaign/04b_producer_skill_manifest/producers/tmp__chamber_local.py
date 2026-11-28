exec(open('/tmp/chamber_probe.py').read().split('for tag,s,h')[0])
from pathlib import Path
Y=v['strain_curves_obs']-v['strain_0'][:,None]; R=v['instant_rate_obs']; n=len(R)
D=m._features(v,v)[:n]@theta[2:]; B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1]])
Q=np.c_[Y,R]; amp=np.linalg.lstsq(B,Q.T,rcond=None)[0].T
for s in (tr,te,v):
 print('panel center/scale by lab',[(round(np.median(s['ref_coupons'][s['lab_id']==l].mean(0)),4),round(np.median(s['ref_coupons'][s['lab_id']==l].std(0)),4)) for l in np.unique(s['lab_id'])]); print('local drift range',np.quantile(m._features(s,s)[:len(s['lab_id'])]@theta[2:],[0,.5,1]))
folds=[v[k]==c for k in ('lab_id','stratum') for c in np.unique(v[k])]
base=np.c_[hv[:,None]*theta[:2],np.zeros(n)]
for name,F,off in [('affine',np.c_[np.ones(n),hv,D],0*base),('hold-weighted',np.c_[hv,D],0*base),('intrinsic hold slopes',np.c_[np.ones(n),D],base),('intrinsic hold slopes, no intercept',D[:,None],base),('stress interaction',np.c_[np.ones(n),hv,D,hv*D],0*base)]:
 def fit(mask):
  return np.linalg.lstsq(F[mask],(Q-off@B.T)[mask],rcond=None)[0]@np.linalg.pinv(B.T)
 E=[]
 for k in folds:
  e=((F[k]@fit(~k)+off[k])@B.T-Q[k]); E.append([np.sqrt(np.mean(e[:,:-1]**2)),np.max(abs(e[:,:-1])),np.sqrt(np.mean(e[:,-1]**2)),np.max(abs(e[:,-1]))])
 print(name,'worst',np.max(E,axis=0).round(6),'coeff',fit(np.ones(n,bool)).tolist())
 if name=='affine': co=fit(np.ones(n,bool))
print('affine residuals sorted by local drift',np.c_[D,hv,np.isnan(v['coupon_features_raw']).sum(1),Q[:,-1]-((np.c_[np.ones(n),hv,D]@co)@B.T)[:,-1]][np.argsort(D)].round(4).tolist())
src=Path('/tmp/hegel_continual_archive/alpha_m100_c02_batch_03_solver.py').read_text(); src=src[:src.index('# The last completed apparatus trace')]
src+='''# Chamber amplitude map from the public verification coupons, using their
# own reference panel. Material parameters are always fitted on the benchtop.
_CHAMBER = np.array('''+repr(co.tolist())+''')


def _deployment(tau, theta, test, hold, times):
    x = _features(test, test)[:len(hold)]
    amplitudes = np.column_stack([np.ones(len(hold)), hold, x @ theta[2:]]) @ _CHAMBER
    basis = np.column_stack([-np.expm1(-times[:, None] / tau), times])
    return amplitudes @ basis.T, amplitudes @ np.r_[1.0 / tau, 1.0]


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    coef = _fit_bridge(dataset['calibration_load_cells'])
    past, hold = _loads(coef, train)
    x = _features(train, test)
    tau, theta = _fit(past, hold, x[:len(hold)], dt, times,
                      train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    hold_test = _invert_bridge(test['hold_channel_raw'], coef).sum(axis=-1)
    strain, rate = _deployment(tau, theta, test, hold_test, times)
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
'''
Path('/holdout_task/batch_solver.py').write_text(src)
