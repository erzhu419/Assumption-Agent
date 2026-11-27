import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d=it['dataset']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); lab=tr['lab_id']; dt,t=d['past_dt'],d['eval_times']; xx=m._prepare(d); x=tuple(a[:n] for a in xx); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; cc=np.isfinite(tr['coupon_features_raw']).all(1)
def report(y,v,r,w,sl):
    return np.round([np.median(abs(y[sl]-v[sl])),np.median(abs(r[sl]-w[sl]))],6)
for fit_cc in (False,True):
    print('complete-only fit',fit_cc)
    for i in np.unique(lab):
        hold=lab==i; fit=~hold & (cc if fit_cc else True)
        p=m._fit(tuple(a[fit] for a in x),dt,t,y[fit],rate[fit],'log')
        a,b,*_=m._design(p[:2],x,dt,t,'log'); print('lab',i,'taus',np.round(np.exp(p[:2]),5),'strain/rate median absolute errors',[(tag,report(y,m._dot(a,p[2:]),rate,m._dot(b,p[2:]),hold&sl).tolist()) for tag,sl in [('active-complete',cc&(x[-1]>0)),('inactive-complete',cc&(x[-1]==0)),('missing',~cc)]])
    fit=cc if fit_cc else np.ones(n,dtype=bool)
    p=m._fit(tuple(a[fit] for a in x),dt,t,y[fit],rate[fit],'log'); print('full refit parameters',np.round(np.r_[np.exp(p[:2]),p[2:]],6))
