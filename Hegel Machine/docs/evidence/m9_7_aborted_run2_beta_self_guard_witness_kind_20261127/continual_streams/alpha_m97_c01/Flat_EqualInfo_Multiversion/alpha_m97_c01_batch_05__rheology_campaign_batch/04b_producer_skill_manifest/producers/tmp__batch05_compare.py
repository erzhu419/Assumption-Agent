import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

path = '/holdout_task/alpha_m97_c01_batch_05_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_05.pkl', 'rb'))[0]
d = item['dataset']
a = d['train']
p,h,x,b = m._inputs(d)
n = len(a['lab_id'])
y = np.column_stack((a['strain_curves_obs'] - b[:n,None], a['instant_rate_obs']))
model = m._Model(d,p,h,x)
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)


def refine(indices, base, mode):
    t,w = base
    target = y[indices]
    r = model.design(t,indices) @ w - target
    scale = 1.4826 * np.median(np.abs(r-np.median(r,axis=0)),axis=0)
    scale = np.maximum(scale, np.median(scale)*0.5)
    if mode == 'pooled':
        scale[:-1] = np.median(scale[:-1])
    for iteration in range(7):
        u = r / (4.685*scale if mode == 'trim' else 2.385*scale)
        if mode == 'trim':
            weight = np.maximum(1-u**2,0) / scale
        else:
            weight = 1 / (scale*np.sqrt(1+u**2))
        weight = weight.ravel()
        ncols = 9 if np.any(model.act[indices]) else 7
        def project(t):
            mat = model.design(t,indices)[:,:,:ncols].reshape(-1,ncols)
            coef = np.linalg.lstsq(mat*weight[:,None],target.ravel()*weight,rcond=None)[0]
            return np.pad(coef,(0,9-ncols)), (mat@coef).reshape(target.shape)-target
        t = least_squares(lambda t: project(t)[1].ravel()*weight,t,bounds=(np.log(.02),np.log(100)),diff_step=1e-4,max_nfev=60,ftol=1e-8).x
        w,r = project(t)
    return t,w

modes = ('base','pooled','per_horizon','trim')
results = {k:np.empty_like(y) for k in modes}
fits = {k:[] for k in modes}
for lab in range(5):
    train = np.flatnonzero(complete & (a['lab_id'] != lab))
    val = np.flatnonzero(a['lab_id'] == lab)
    base = m._fit(model,y,train)
    for mode in modes:
        t,w = base if mode == 'base' else refine(train,base,mode)
        results[mode][val] = model.design(t,val)@w-y[val]
        fits[mode].append(np.exp(t))
        if lab == 4:
            z = model.design(t,slice(n,None))@w
            e = z-np.column_stack((item['gt']['te_clean']-b[n:,None],item['gt']['te_true_rate0']))
            print(mode,'full tau',np.exp(t),'weights',np.round(w,5),'public RMSE',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)))
for mode,r in results.items():
    scores=[]
    for lab in range(4):
        for gate in (False,True):
            ix=complete&(a['lab_id']==lab)&((model.act[:n]>0)==gate)
            c=r[ix,:-1]; rate=r[ix,-1]
            scores.append([np.sqrt(np.mean(np.minimum(abs(c),.10)**2)),np.sqrt(np.mean(rate**2)),np.median(abs(c))])
    print(mode,'LOLO taus',np.round(fits[mode][:4],4),'mean and worst (clipped curve RMSE, rate RMSE, curve MAE median)',np.round(np.mean(scores,axis=0),5),np.round(np.max(scores,axis=0),5))
