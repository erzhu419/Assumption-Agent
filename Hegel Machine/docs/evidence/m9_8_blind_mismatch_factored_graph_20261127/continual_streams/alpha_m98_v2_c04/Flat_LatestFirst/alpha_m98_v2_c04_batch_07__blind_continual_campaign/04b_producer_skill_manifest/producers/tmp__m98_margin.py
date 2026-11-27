import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
ds, q = item['dataset'], item['campaign_diagnostics']
o = m.solve_batch(ds); tau, th = o['estimated_taus'], o['hereditary_weights']
past, h = m._loads(q, m._fit_bridge(ds['calibration_load_cells']))
x = m._features(q,q)[:len(h)]; d = th[2]+x@th[3:]
t, dt = ds['eval_times'], ds['past_dt']
later = np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
y = np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]
lab, stratum = q['lab_id'], q['stratum']; allrows = np.ones(len(h), dtype=bool)

def make(tt):
    z = past @ (-np.expm1(-dt[:,None]/tt)*np.exp(-later[:,None]/tt))
    return h[:,None]-z, np.vstack((np.c_[-np.expm1(-t[:,None]/tt),t],np.r_[1/tt,1.]))

def fit(ix, mode):
    def design(p):
        g,b = make(np.exp(p[:2])); fast, slow = g.T
        u,v,w = fast[:,None], np.c_[slow,d], np.c_[np.ones(len(h)),h,d]
        if mode == 'fast_difference': u = np.c_[fast, slow-fast]
        if mode == 'slow_difference': v = np.c_[slow,d,slow-fast]
        if mode == 'steady_difference': w = np.c_[w,slow-fast]
        if mode == 'slow_product': v = np.c_[v,slow*d]
        if mode == 'slow_rational': v = np.c_[slow,d/(1+p[2]*d*d)]
        if mode == 'steady_rational': w = np.c_[np.ones(len(h)),h,d/(1+p[2]*d*d)]
        return np.concatenate([f[:,None,:]*b[None,:,j,None] for j,f in enumerate((u,v,w))],axis=2)
    par = np.log(tau*m._CHAMBER_TAU_SCALE)
    lo, hi = list(np.log([.25,.65])), list(np.log([.65,8.]))
    if mode.endswith('rational'): par=np.r_[par,.1]; lo.append(0.); hi.append(5.)
    k=len(par); a=design(par)[ix]; c=np.linalg.lstsq(a.reshape(-1,a.shape[-1]),y[ix].ravel(),rcond=None)[0]
    f=least_squares(lambda p: ((design(p[:k])@p[k:]-y)[ix]).ravel(),np.r_[par,c],bounds=(np.r_[lo,np.full(len(c),-np.inf)],np.r_[hi,np.full(len(c),np.inf)]),loss='soft_l1',f_scale=.012,max_nfev=140)
    return design(f.x[:k])@f.x[k:]-y

def metric(e):
    return np.array([np.mean(abs(e[:,:-1])),np.mean(abs(e[:,-1])),np.quantile(abs(e[:,:-1]),.9),np.quantile(abs(e[:,-1]),.9)])

modes=['base','fast_difference','slow_difference','steady_difference','slow_product','slow_rational','steady_rational']
# Select challengers only on inner folds. Judge them on disjoint outer labs/strata.
for outer in (lab,stratum):
    base, challenger = np.zeros_like(y),np.zeros_like(y)
    for val in np.unique(outer):
        out=outer==val; keep=~out; ranking=[]
        for mode in modes:
            inner=np.zeros_like(y)
            for j in np.unique(outer[keep]):
                v=keep&(outer==j); inner[v]=fit(keep&~v,mode)[v]
            a=metric(inner[keep]); ranking.append((max(a[2:]),mode))
        _,selected=min(ranking); base[out]=fit(keep,'base')[out]; challenger[out]=fit(keep,selected)[out]
        print('outer', 'lab' if outer is lab else 'stratum',val,selected,'base',metric(base[out]).round(5),'selected',metric(challenger[out]).round(5))
    groups=[outer==j for j in np.unique(outer)]
    for name,e in [('base',base),('selected',challenger)]:
        print(name,'pooled',metric(e).round(5),'worst-slice',np.max([metric(e[g]) for g in groups],axis=0).round(5))
# Inspect signed OOF residuals by lab/stratum to expose cancellation hidden by pooling.
for group_name, groups in [('lab',lab),('stratum',stratum)]:
    for j in np.unique(groups):
        val=groups==j; e=fit(~val,'base')[val]
        print('signed',group_name,j,np.median(e[:,:-1]).round(5),np.median(e[:,-1]).round(5))
