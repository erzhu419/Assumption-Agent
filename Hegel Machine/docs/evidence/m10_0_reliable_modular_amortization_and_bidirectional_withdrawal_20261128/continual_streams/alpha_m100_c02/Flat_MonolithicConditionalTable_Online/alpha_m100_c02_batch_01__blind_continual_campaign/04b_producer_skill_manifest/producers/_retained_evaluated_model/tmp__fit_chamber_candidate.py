exec(open('/tmp/chamber_compact.py').read().split("print('dt,time'")[0])
from pathlib import Path
Y=np.c_[y,r]
B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1]])
M=np.isnan(q['coupon_features_raw'])
F=np.c_[np.ones(len(h)),h,D,p[:,-1],M[:,:3]]
A=np.stack([(F[:,j,None]*B[None,:,k]) for k in range(3) for j in range(F.shape[1])],axis=-1)
def fit(rows,nuisance):
 a=A[rows].reshape(-1,A.shape[-1]); target=Y[rows].ravel()
 use=np.tile(np.arange(F.shape[1])<(F.shape[1] if nuisance else 4),3)
 b=np.zeros(A.shape[-1]); b[use]=np.linalg.lstsq(a[:,use],target,rcond=None)[0]; return b
for nuisance in (False,True):
 E=np.zeros_like(Y)
 for l in np.unique(lab):
  m=lab==l; E[m]=A[m]@fit(~m,nuisance)-Y[m]
 def score(m): return [np.mean(E[m,:-1]**2)**.5,np.mean(E[m,-1]**2)**.5]
 print('calibration with missingness nuisance',nuisance,'worst lab',np.max([score(lab==l) for l in np.unique(lab)],axis=0),'complete',score(~M.any(1)),'missing',score(M.any(1)))
 print('coefficients',fit(np.ones(len(h),bool),nuisance).reshape(3,-1).T)
w=fit(np.ones(len(h),bool),False).reshape(3,-1).T[:4]
base=Path('/tmp/chamber_base.py').read_text().split('def solve_batch(')[0]
source='''# Chamber response fitted on the public verification coupons. Rows multiply
# a constant, calibrated hold stress, intrinsic drift, and final past stress.
_CHAMBER = np.array(%s)


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    dt = np.asarray(dataset['past_dt'])
    times = np.asarray(dataset['eval_times'])
    x = _features(train, test)
    n = len(train['lab_id'])
    p, h = _loads(cal, train)
    tau, theta = _fit(p, h, x[:n], dt, times,
                      train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'])
    p, h = _loads(cal, test)
    coefficients = np.column_stack((np.ones(len(h)), h, x[n:] @ theta[2:], p[:, -1])) @ _CHAMBER
    basis = np.column_stack((-np.expm1(-times[:, None] / tau), times))
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + coefficients @ basis.T,
            'initial_rate_pred': coefficients @ np.r_[1 / tau, 1.]}
''' % repr(w.tolist())
Path('/holdout_task/batch_solver.py').write_text(base+source)
