import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w03_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_w03_batch_03.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
a=d['train']; n=len(a['lab_id']); x,h,p=[z[:n] for z in m._prep(d)]; dt,t=d['past_dt'],d['eval_times']; lab=a['lab_id']; mem=a['initial_memory_states']; rate=a['instant_rate_obs']; y=a['strain_curves_obs']-a['strain_0'][:,None]; miss=np.isnan(a['coupon_features_raw']).any(1)
z=x[:,1:]; i,j=np.triu_indices(z.shape[1],1); linear=z.copy(); extras={'pair differences':np.column_stack((z[:,i]-z[:,j],z[:,i]+z[:,j])), 'quadratic and cross-products':np.column_stack((z*z,z[:,i]*z[:,j])), 'three-way products':np.column_stack([np.prod(z[:,np.arange(4)!=k],axis=1) for k in range(4)]), 'rational corrections':np.column_stack([z/(1+c*z*z)-z for c in (.25,1.)])}
out={k:[] for k in ('linear',*extras,'complete_rate_blend','missing_baseline','missing_rate_blend')}
for l in np.unique(lab):
 f=lab!=l; s=~f; tau,b=m._fit(x[f],h[f],p[f],dt,t,mem[f],y[f],rate[f]); c,r,_,_=m._design(tau,x,h,p,dt,t,mem)
 # Remove the two hereditary terms and estimate individual drift from all horizons using a robust location fit.
 ratios=(y-c[:,:,:2]@b[:2])/t; drift=np.median(ratios[:,len(t)//2:],axis=1); innovation=rate-r@b; baseline=x@b[2:]; cm=s&~miss; mm=s&miss; fc=f&~miss
 mae=lambda pred,mask:float(np.mean(abs(pred[mask]-drift[mask])))
 out['linear'].append(mae(baseline,cm));out['complete_rate_blend'].append(mae(baseline+.25*innovation,cm));out['missing_baseline'].append(mae(baseline,mm));out['missing_rate_blend'].append(mae(baseline+.95*innovation,mm))
 for k,v in extras.items():
  if np.linalg.matrix_rank(np.c_[x[fc],v[fc]])==np.linalg.matrix_rank(x[fc]): out[k].append(out['linear'][-1]); continue
  fit=least_squares(lambda w:(v[fc]@w-(drift-baseline)[fc])/.008,np.zeros(v.shape[1]),loss='soft_l1').x
  out[k].append(mae(baseline+v@fit,cm))
for k,v in out.items(): print(k,'lab MAE',np.round(v,6),'worst',round(max(v),6))
print('train response fields',list(a)); print('test response fields',list(d['test'])); print('upsweep yield headroom',d['yield_stress']-max(np.max(abs(p)),np.max(abs(h))))
