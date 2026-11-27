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
    if sel.any(): print(name,sel.sum(),'curve/rate MAE',np.mean(abs(pc[sel]-y[sel])),np.mean(abs(pr[sel]-r[sel])))
e=pc-y; f=pr-r
for lab in range(4):
    for stratum in np.unique(tr['stratum']):
        s=(tr['lab_id']==lab)&(tr['stratum']==stratum)
        if s.any():
            print('cell',lab,stratum,'n',s.sum(),'median signed curve/rate error',np.median(e[s]),np.median(f[s]),'median absolute',np.median(abs(e[s])),np.median(abs(f[s])))
print('absolute curve residual quantiles',np.quantile(abs(e),[.5,.9,.95,.99,.995,1]))
print('absolute rate residual quantiles',np.quantile(abs(f),[.5,.9,.95,.99,1]))
print('curve residuals above 0.5 by time',np.sum(abs(e)>.5,axis=0),'affected coupons',np.sum(np.any(abs(e)>.5,axis=1)))
cal=d['calibration_load_cells']; base=m._calibrate(cal)
for lab in range(4):
    u=np.flatnonzero(cal['lab_id']==lab); z=np.argsort(cal['cmd_stress'][u]); picks=u[z[[0,len(z)//2,-1]]]; s=(cal['lab_id']!=lab)|np.isin(np.arange(len(cal['lab_id'])),picks)
    err=[]
    for i in picks:
        q={k:v[s].copy() for k,v in cal.items()}; q['true_stress'][np.flatnonzero(np.flatnonzero(s)==i)[0]]+=4.; err.append(np.max(abs(m._calibrate(q)[lab]-base[lab])))
    print('3-point sweep, one EMI spike, max errors',lab,err)
