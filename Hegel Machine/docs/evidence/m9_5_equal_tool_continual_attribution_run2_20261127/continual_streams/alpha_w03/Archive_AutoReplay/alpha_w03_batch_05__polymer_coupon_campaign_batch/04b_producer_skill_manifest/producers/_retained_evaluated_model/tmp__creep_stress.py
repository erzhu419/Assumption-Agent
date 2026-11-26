import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_w03_batch_05_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_alpha_w03_batch_05.pkl','rb'))[0]; d,g=a['dataset'],a['gt']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; out=m.solve_alpha_w03_batch_05(d)
e=out['predicted_strain']-g['te_clean']; r=out['initial_rate_pred']-g['te_true_rate0']; rmse=lambda q: np.sqrt(np.mean(q*q)); print('public worst horizon RMSE',max(rmse(v) for v in e.T),'worst stratum strain/rate',max(rmse(e[te['stratum']==j]) for j in np.unique(te['stratum'])),max(rmse(r[te['stratum']==j]) for j in np.unique(te['stratum'])))
assert np.array_equal(out['viscoplastic_weights'],np.zeros(2))
X,h,p=m._prep(d); n=len(tr['lab_id']); good=np.isfinite(tr['coupon_features_raw']).all(axis=1); sy=.85; tau0=np.array([.64,3.25]); coef=np.array([.7,.48,.08,.15,-.12,.11,-.09,.051,.023]); hist=np.zeros((len(h),2))
for i,step in enumerate(dt):
    decay=np.exp(-step/tau0); hist=decay*hist+(1-decay)*p[:,i,None]
gap=(h[:,None]-hist)*coef[:2]; oh=np.maximum(abs(h)-sy,0)**2*np.sign(h); work=np.sum(np.maximum(abs(p)-sy,0)**2*dt,axis=1); drift=X@coef[2:7]+coef[7]*oh; gap[:,0]+=coef[8]*work*h
Y=gap@(1-np.exp(-t[None,:]/tau0[:,None]))+drift[:,None]*t; R=gap@(1/tau0)+drift
q=np.flatnonzero(good); z=np.arange(n)[q]; rng=np.random.default_rng(82); yy=Y[z].copy(); spike=rng.random(yy.shape)<.1; yy[spike]+=rng.choice([-1.,1.],size=spike.sum()); tau,b=m._fit(X[z],h[z],p[z],dt,t,sy,yy,R[z]); c,f=m._design(tau,X[n:],h[n:],p[n:],dt,t,sy)
for name,mask in [('positive hold',h[n:]>sy),('negative hold',h[n:]< -sy),('history only',(abs(h[n:])<=sy)&(work[n:]>0))]:
    assert mask.any(); ce=rmse((c@b-Y[n:])[mask]); re=rmse((f@b-R[n:])[mask]); print('synthetic',name,int(mask.sum()),ce,re); assert max(ce,re)<.001
print('synthetic parameter errors',max(abs(tau-tau0)),max(abs(b-coef))); assert max(abs(tau-tau0))<.01 and max(abs(b-coef))<.001
