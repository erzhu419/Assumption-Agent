import importlib.util, pickle
import numpy as np
path='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',path+'alpha_m96_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
a=pickle.load(open(path+'public_alpha_m96_w02_s00.pkl','rb'))[0]['dataset']; tr,te=a['train'],a['test']; c=a['calibration_load_cells']; labs=c['lab_id']; gain,bias=s._calibration(c)
for lab in range(4):
 ids=np.flatnonzero(labs==lab); err=[]
 for i in ids:
  keep=np.arange(len(labs))!=i; g,b=s._calibration({k:np.asarray(c[k])[keep] for k in ('lab_id','cmd_stress','true_stress')}); err.append(g[lab]*c['cmd_stress'][i]+b[lab]-c['true_stress'][i])
 print('spot jackknife lab',lab,'median/max abs stress error',np.median(np.abs(err)),max(np.abs(err)))
t=a['eval_times']; dt=a['past_dt']; lab=tr['lab_id']; st=tr['stratum']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); design=s._inputs(tr,s._align(tr,te)[:len(lab)],gain,bias); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; cells=[]
for j in np.unique(lab):
 keep=lab!=j; v=(~keep)&complete; tau,w=s._fit(dt,t,tuple(z[keep] for z in design),y[keep],r[keep]); history,_,onset,_=s._basis(tau,dt,t,design[0][v],design[1][v]); drift=design[2][v]@w[2:]; e=y[v]-history@w[:2]-drift[:,None]*t; q=r[v]-onset@w[:2]-drift; sv=st[v]; pos=t>0
 for k in np.unique(sv):
  test=sv==k; ref=~test; offset=np.median(q[ref]); independent=np.median(np.median(e[ref][:,pos]/t[pos],axis=1)); m=lambda z:float(np.median(np.abs(z)))
  cells.append([m(e[test]),m(e[test]-offset*t),m(q[test]),m(q[test]-offset)]); print('hold lab/stratum',j,k,'n',test.sum(),'calibration offsets onset/curve',round(offset,5),round(independent,5),'baseline/corrected median abs residuals',np.round(cells[-1],5))
print('worst complete-cell medians strain baseline/corrected, rate baseline/corrected',np.max(cells,axis=0))
