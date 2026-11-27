import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m96_w04_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m96_w04_batch_01.pkl', 'rb'))[0]['dataset']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
x = m._aligned(tr, te)[:n]
labs = tr['lab_id']
cal = m._calibration(d['calibration_load_cells'])
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None], tr['instant_rate_obs']]
times = d['eval_times']
clock = np.r_[times, 1.]
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)

def model(mask, zero=False):
    part = {k:v[mask] for k,v in tr.items()}
    if zero:
        part['initial_memory_states'] = np.zeros_like(part['initial_memory_states'])
    return m._Creep(part, x[mask], cal, d['past_dt'], times)

# Quadratics include the span of squared pairwise sums and differences.
quad = np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
cubic = np.column_stack([x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4)])
rat = np.column_stack([x[:,i]/(1+x[:,j]**2) for i in range(4) for j in range(4)])
mods = {'baseline':None, 'quadratic':quad, 'three_way':cubic, 'rational':rat}
oof = {k:np.empty_like(y) for k in [*mods,'zero_preload']}
exact = 0.
for lab in np.unique(labs):
    train = labs != lab
    val = ~train
    a, b = model(train), model(val)
    tau, theta = m._fit(a,y[train])
    A, B = a.basis(tau)[0], b.basis(tau)[0]
    baseline = A @ theta - y[train]
    scale = np.r_[np.full(len(times), max(2*m._mad(baseline[:,:-1]),1e-6)), max(2*m._mad(baseline[:,-1]),1e-6)]
    oof['baseline'][val] = B @ theta-y[val]
    for name, extra in mods.items():
        if extra is None:
            continue
        center = extra[train].mean(axis=0)
        spread = extra[train].std(axis=0)
        e = ((extra-center)/np.maximum(spread, 1e-8))[:,None,:]*clock[None,:,None]
        AA = np.concatenate((A,e[train]),axis=-1)
        BB = np.concatenate((B,e[val]),axis=-1)
        valid = np.isfinite(y[train])
        s = np.broadcast_to(scale,y[train].shape)[valid]
        X, Y = AA[valid]/s[:,None], y[train][valid]/s
        opt = least_squares(lambda p:X@p-Y,np.r_[theta,np.zeros(extra.shape[1])],jac=lambda p:X.copy(),loss='cauchy',x_scale='jac',max_nfev=80)
        oof[name][val] = BB@opt.x-y[val]
    no_a,no_b = model(train,True),model(val,True)
    nt, nw = m._fit(no_a,y[train])
    oof['zero_preload'][val] = no_b.basis(nt)[0]@nw-y[val]
    # Independent sequential recurrence compared to closed-form basis for both measured memory slots.
    for slot,t in enumerate(tau):
        z = b.initial[:,slot].copy()
        for j,dt in enumerate(d['past_dt']):
            z += (b.past[:,j]-z)*(-np.expm1(-dt/t))
        expected = (b.hold-z)[:,None] * np.r_[-np.expm1(-times/t),1/t]
        exact = max(exact,np.max(abs(expected-B[:,:,slot])))

for name, e in oof.items():
    a = abs(e)
    bylab = [np.nanmedian(a[labs==l,:-1]) for l in range(4)]
    bywing = [np.nanmedian(a[mask,:-1]) for mask in (labs<2,labs>=2)]
    worst_stratum = max(np.nanmedian(a[(labs==l)&(tr['stratum']==s),:-1]) for l,s in np.unique(np.c_[labs,tr['stratum']],axis=0))
    print(name,'per-lab curve MedAE',np.round(bylab,6),'per-wing',np.round(bywing,6),'worst lab-stratum',round(worst_stratum,6),'complete final/rate',np.round(np.nanmedian(a[complete,-2:],axis=0),6),'missing curve MedAE',round(np.nanmedian(a[~complete,:-1]),6))
print('sequential vs closed-form max difference',exact)
assert exact < 1e-10
