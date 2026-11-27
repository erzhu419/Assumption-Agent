import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py')
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; q=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']
trc,tec=m['_prepare'](d); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; idx=np.arange(len(y)); cal=d['calibration_load_cells']; raw=cal['bridge_raw']; target=cal['true_channel_loads'].sum(1)
lin=np.c_[np.ones(len(raw)),raw]; pair=np.column_stack([raw[:,i]*raw[:,j] for i in range(4) for j in range(i,4)]); quad=np.c_[lin,pair]; order=np.argsort(target); folds=np.array_split(order,5)
for name,X in [('affine',lin),('quadratic + channel interactions',quad)]:
 e=[]
 for fold in folds:
  k=np.setdiff1d(np.arange(len(raw)),fold); a=np.linalg.lstsq(X[k],target[k],rcond=None)[0]; e.append(abs(X[fold]@a-target[fold]));
 print('bridge',name,'worst stress-block P90',max(np.quantile(a,.9) for a in e))

def fit(c, y, r, mode):
 tau, th=m['_fit'](dt,t,c,y,r)
 if mode=='current': return tau,th
 k=y.size
 def err(a):
  C,R=m['_design'](np.exp(a[:2]),dt,t,c); return np.r_[(C@a[2:]-y).ravel(),R@a[2:]-r]
 a=np.r_[np.log(tau),th]; e=err(a); noise=[1.4826*np.median(abs(v-np.median(v))) for v in (e[:k],e[k:])]; s=np.repeat(np.maximum(noise,1e-6),[k,len(r)]); loss,curve,cut=mode
 s[:k]*=curve
 a=least_squares(lambda a:err(a)/s,a,bounds=([np.log(.050001)]*2+[-np.inf]*7,[np.log(24.9999)]*2+[np.inf]*7),loss=loss,f_scale=cut,x_scale='jac').x
 j=np.argsort(a[:2]); return np.exp(a[j]),np.r_[a[2+j],a[4:]]
for mode in ['current',('soft_l1',1,1),('cauchy',1,1.5),('cauchy',1.7,2.5)]:
 tau,th=fit(trc,y,r,mode); stats=[]; masks=[]
 for name in ['lab_id','stratum']:
  for lab in np.unique(tr[name]): masks.append(tr[name]==lab)
 for k in masks:
  a,b=fit(tuple(x[~k] for x in trc),y[~k],r[~k],mode); C,R=m['_design'](a,dt,t,tuple(x[k] for x in trc)); e=abs(C@b-y[k]); v=abs(R@b-r[k]); stats.append([np.quantile(e,.9),np.quantile(v,.9),a[1],np.quantile(e[:,-3:],.9)])
 stats=np.array(stats); print('material',mode,'full tau',tau.round(5),'held-out worst strain/rate/late',stats[:,[0,1,3]].max(0).round(5),'tau1 range',stats[:,2].min(),stats[:,2].max())
