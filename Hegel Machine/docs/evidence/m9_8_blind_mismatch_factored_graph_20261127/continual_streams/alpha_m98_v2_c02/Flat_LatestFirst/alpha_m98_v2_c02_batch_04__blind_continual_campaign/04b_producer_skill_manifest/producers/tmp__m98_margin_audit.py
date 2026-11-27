import importlib.util, pickle
from itertools import combinations, combinations_with_replacement
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']; n = len(x)
cols = np.arange(4); adjacent = (cols+1)%4
base = np.c_[np.ones(n), x]
sparse = [np.c_[base, x[:,j]**3, x[:,j]*x[:,adjacent[j]]] for j in cols]
def fit(A, y, mask):
    return A @ np.linalg.lstsq(A[mask], y[mask], rcond=None)[0]
mixed = np.count_nonzero(x, axis=1)>1
pilot = np.column_stack([fit(a, y[:,j], mixed) for j,a in enumerate(sparse)])
e = np.linalg.norm(y-pilot, axis=1); med = np.median(e)
clean = e <= med+10*1.4826*np.median(abs(e-med))
print('clean and rejected sweep rows:', clean.sum(), np.flatnonzero(~clean), 'minimum rejected / maximum clean raw residual:', e[~clean].min(), e[clean].max())
poly = np.c_[base, np.column_stack([np.prod(x[:,i], axis=1) for degree in (2,3) for i in combinations_with_replacement(range(4),degree)])]
contrasts = np.column_stack([x[:,i]+sgn*x[:,j] for i,j in combinations(cols,2) for sgn in (-1,1)])
q = np.c_[x, contrasts, x.sum(axis=1)]
rational = q/(1+q*q)
extended = [np.c_[a, rational] for a in sparse]
axis = np.argmax(abs(x),axis=1)
blocks = [(f'axis-{j}', clean & (np.count_nonzero(x,axis=1)==1) & (axis==j)) for j in cols]
blocks += [('mixed-high', clean & mixed & (np.linalg.norm(x,axis=1)>=np.quantile(np.linalg.norm(x[mixed],axis=1),.75)))]
for name, held in blocks:
    if not held.any(): continue
    keep = clean & ~held
    rates = []
    for models in (sparse,[poly]*4,extended):
        pred = np.column_stack([fit(a,y[:,j],keep) for j,a in enumerate(models)])
        rates.append(np.max(abs(pred[held]-y[held])))
    cal_keep = ~held
    f=m._stress_calibration({'bridge_raw':y[cal_keep], 'true_channel_loads':x[cal_keep]})
    inv=np.max(abs(f(y[held])-x[held].sum(axis=1)))
    aff=fit(np.c_[np.ones(n),y],x.sum(axis=1),keep)
    print(name,'n=', held.sum(),'max raw sparse/full-cubic/rational:',np.round(rates,7),'max total sparse/affine:',inv,np.max(abs(aff[held]-x[held].sum(axis=1))))
    assert inv < .002
train,test=d['train'],d['test']; u,_=m._standardize(train,test); f=m._stress_calibration(c)
design=m._design(train,u,f,d['past_dt'],d['eval_times']); s=train['strain_curves_obs']-train['strain_0'][:,None]; r=train['instant_rate_obs']; lab=train['lab_id']; aux=train['aux_chamber_telemetry']
for j in np.unique(lab):
    use=lab!=j; held=~use
    tau,theta=m._fit(lambda ts:tuple(a[use] for a in design(ts)),s[use],r[use]); A,B=design(tau)
    es=s-A@theta; er=r-B@theta
    ds=np.median(es,axis=1); dr=er
    # Estimate a telemetry slope after removing training-lab means; do not fit an offset on the held lab.
    v=aux.copy(); h=np.column_stack([ds,dr]); w=h.copy()
    for k in np.unique(lab[use]):
        mask=lab==k; v[mask]-=v[mask].mean(0); w[mask]-=w[mask].mean(0)
    b=np.linalg.lstsq(v[use],np.clip(w[use],-.05,.05),rcond=None)[0]
    adjusted=h[held]-(aux[held]-aux[use].mean(0))@b
    print('lab',j,'signed residual medians strain/rate:',np.round(np.median(h[held],axis=0),6),'P90 before/after telemetry:',np.round(np.quantile(abs(h[held]),.9,axis=0),6),np.round(np.quantile(abs(adjusted),.9,axis=0),6))
