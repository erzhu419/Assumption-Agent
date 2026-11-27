import importlib.util,pickle,numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_m97_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open(root+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']; tr=d['train']; n=len(tr['lab_id']); data=tuple(x[:n] for x in m._prepare(d)); dt=d['past_dt']; t=d['eval_times']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; pc=np.empty_like(y); pr=np.empty_like(r)
for lab in range(4):
    sel=tr['lab_id']==lab
    tau, w=m._fit(tuple(x[~sel] for x in data),dt,t,y[~sel],r[~sel]); c,b=m._design(tuple(x[sel] for x in data),dt,t,tau); pc[sel]=c@w; pr[sel]=b@w
    print('lab',lab,'tau',tau,'curve/rate RMSE',np.mean((pc[sel]-y[sel])**2)**.5,np.mean((pr[sel]-r[sel])**2)**.5,'MAE',np.mean(abs(pc[sel]-y[sel])),np.mean(abs(pr[sel]-r[sel])))
print('horizon RMSE',np.mean((pc-y)**2,axis=0)**.5)
for name,sel in [('negative hold',data[1]<0),('positive hold',data[1]>=0),('missing coupon',np.isnan(tr['coupon_features_raw']).any(axis=1)),('above critical stress',np.abs(data[1])>d['crit_stress'])]:
    print(name,sel.sum(),'curve/rate MAE',np.mean(abs(pc[sel]-y[sel])),np.mean(abs(pr[sel]-r[sel])))
out=m.solve_alpha_m97_c01_batch_03(d)
for k in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(k,out[k])
