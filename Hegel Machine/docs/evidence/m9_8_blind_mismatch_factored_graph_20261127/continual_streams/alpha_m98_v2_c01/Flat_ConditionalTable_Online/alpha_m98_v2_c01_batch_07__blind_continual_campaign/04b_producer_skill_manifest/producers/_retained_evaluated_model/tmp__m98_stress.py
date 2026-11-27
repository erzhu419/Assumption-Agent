import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; cal=d['calibration_load_cells']; x=cal['true_channel_loads']; y=cal['bridge_raw']; clean=np.max(np.abs(y-np.median(y,axis=0)),axis=1)<8; x,y=x[clean],y[clean]; order=np.argsort(np.max(abs(x),axis=1)); high=order[int(.8*len(x)):]; split=[(f'fold{k}', h) for k,h in enumerate(np.array_split(np.random.default_rng(72).permutation(len(x)),5))]+[('high load',high)]
def design(x,j,kind,a=0.0):
 u,v=x[:,j],x[:,(j+1)%4]; cols=[np.ones(len(x)),*x.T,u*v,u**3]; q=u-v
 if kind=='pairwise': cols += [u*u,v*v,q**3,u*u*v,u*v*v]
 if kind=='rational': cols[5]=cols[5]/(1+a*q*q); cols[6]=cols[6]/(1+a*q*q)
 return np.column_stack(cols)
for name,h in split:
 keep=np.ones(len(x),bool); keep[h]=False; sub={'bridge_raw':y[keep],'true_channel_loads':x[keep]}; coeff=m._axial_load_map(sub); spec=m._specimen({'past_channel_raw':y[h,None,:],'hold_channel_raw':y[h]},np.zeros((len(h),4)),coeff)
 print(name,'production axial RMSE',np.sqrt(np.mean((spec[1]-x[h].sum(axis=1))**2)))
 for kind in ('base','pairwise','rational'):
  errors=[]
  for j in range(4):
   def fit(a): return np.linalg.lstsq(design(x[keep],j,kind,a),y[keep,j],rcond=None)[0]
   a=least_squares(lambda z:design(x[keep],j,kind,z[0])@fit(z[0])-y[keep,j],[.01],bounds=(0,5)).x[0] if kind=='rational' else 0.
   errors.append(design(x[h],j,kind,a)@fit(a)-y[h,j])
  print(' ',kind,'forward RMSE',np.sqrt(np.mean(np.square(errors))))
tr,te,diag=d['train'],d['test'],i['campaign_diagnostics']; t,dt=d['eval_times'],d['past_dt']; xt,_=m._features(tr,te); empty={k:diag[k][:0] for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; xd,_=m._features(diag,empty); bridge=m._axial_load_map(cal); src=m._specimen(tr,xt,bridge); dst=m._specimen(diag,xd,bridge); ytrain=tr['strain_curves_obs']-tr['strain_0'][:,None]; yc=diag['strain_curves_obs']-diag['strain_0'][:,None]; yr=diag['instant_rate_obs']; ec=np.empty_like(yc); er=np.empty_like(yr)
for lab in np.unique(diag['lab_id']):
 keep=tr['lab_id']!=lab; h=diag['lab_id']==lab; tau,theta=m._fit(tuple(a[keep] for a in src),dt,t,ytrain[keep],tr['instant_rate_obs'][keep]); pc,pr=m._chamber_predict(dst,dt,t,tau,theta); ec[h]=pc[h]-yc[h]; er[h]=pr[h]-yr[h]
for name,labels in [('feature missingness',np.isnan(diag['coupon_features_raw']).any(axis=1)),('stratum',diag['stratum'])]:
 for v in np.unique(labels):
  h=labels==v; print(name,v,'count',h.sum(),'curve RMSE and bias by horizon',[(round(np.sqrt(np.mean(z*z)),6),round(np.mean(z),6)) for z in np.array_split(ec[h],3,axis=1)],'rate RMSE and bias',np.sqrt(np.mean(er[h]**2)),np.mean(er[h]))
