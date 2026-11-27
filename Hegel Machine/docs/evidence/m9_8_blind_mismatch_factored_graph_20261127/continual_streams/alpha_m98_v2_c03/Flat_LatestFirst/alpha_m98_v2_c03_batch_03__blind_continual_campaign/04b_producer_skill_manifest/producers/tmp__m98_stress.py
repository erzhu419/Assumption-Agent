import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/batch_solver.py'; q=importlib.util.spec_from_file_location('solver',p); s=importlib.util.module_from_spec(q);q.loader.exec_module(s)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr=d['train']; g=i['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; train,test=s._prepare(d); o,c=s._calibration(d['calibration_load_cells'])
def observe(a): return np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']]
def select(coupon,m): return tuple(z[m] for z in coupon)
def scores(e):return np.sqrt([np.mean(e[:,:-1]**2),np.mean(e[:,-2]**2),np.mean(e[:,-1]**2)])
y=observe(tr); lab=tr['lab_id']; T,w=s._fit(train,dt,t,y)
for kind in ('cauchy','linear'):
 rows=[]
 for l in np.unique(lab):
  m=lab==l; tt,ww=s._fit(select(train,~m),dt,t,y[~m],loss=kind); e=s._design(select(train,m),dt,t,tt)@ww-y[m]; rows.append(np.r_[scores(e),np.median(abs(e[:,:-1])),np.max(abs(tt/T-1)),np.max(abs(ww-w))]);
 print('train holdout per lab',kind,np.round(rows,5))
x=g['coupon_features_raw'].copy(); lab=g['lab_id']; st=g['stratum']; yg=observe(g); raw=x.copy()
for l in np.unique(lab):
 m=lab==l; a=g['ref_coupons'][m]; x[m]=(x[m]-np.median(a.mean(0)))/np.median(a.std(0))
p=(g['past_channel_raw']-o)@c; h=(g['hold_channel_raw']-o)@c
for kind in ('unit','affine_rational','pair_interaction'):
 rows=[]; params=[]
 for l in np.unique(lab):
  m=lab==l; j=~m; _,iv,n=np.unique(np.c_[lab[j],st[j]],axis=0,return_inverse=True,return_counts=True); v=1/n[iv].reshape(-1,1); mu=np.nansum(v*x[j],0)/np.sum(v*~np.isnan(x[j]),0); xx=np.where(np.isnan(x),mu,x); X=np.c_[np.ones(len(x)),xx]; a=s._design((p,h,X),dt,t,T); b=a[:,:,:2]@w[:2]; r=X@w[2:]; f=np.r_[-np.expm1(-t),1.]
  if kind=='unit': pred=b+r[:,None]*f; z=[]
  else:
   def model(z):
    if kind=='affine_rational': u=(r+z[0])/(1+z[1]*r*r)
    else: u=r+z[0]*(p[:,-1]-p[:,-2])+z[1]*(r*h*p[:,-1])
    return b+u[:,None]*f
   z=least_squares(lambda q:(model(q)[j]-yg[j]).ravel(),[0,0],bounds=([-np.inf,0] if kind=='affine_rational' else [-np.inf,-np.inf],[np.inf,np.inf])).x; pred=model(z)
  rows.append(scores(pred[m]-yg[m])); params.append(z)
  if kind=='unit':
   for missing in (False,True):
    k=m & (np.isnan(raw).any(1)==missing)
    if k.any(): print('diagnostic slice lab/missing/count',l,missing,k.sum(),np.round(scores(pred[k]-yg[k]),5))
 print('diagnostic worst lab',kind,np.round(np.max(rows,0),5),'fold parameters',np.round(params,4))
