import importlib.util, pickle
import numpy as np
r='/holdout_task/'
s=importlib.util.spec_from_file_location('m',r+'alpha_w01_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
c=pickle.load(open(r+'public_alpha_w01_batch_07.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']; t,dt,sy=(d[k] for k in ('eval_times','past_dt','yield_stress')); mp=m._calibrate(d['calibration_load_cells']); x,xt,co,ct=m._align(tr,te); lab=tr['lab_id']; fold=np.load('/tmp/b07_oof.npz')['fold']
def pred(b,xx,ts,w):
 h,v=m._stress(b,mp); A,B=m._design(ts,h,v,xx,dt,t,*m._plastic(h,v,dt,sy)); return b['strain_0'][:,None]+A@w,B@w
def slope(curve,observed):
 z=(observed-curve)[:,t>=np.median(t)]/t[t>=np.median(t)]; return np.median(z,axis=1)
def correction(z,err):
 A=np.column_stack((np.ones(len(z)),z)); w=np.ones(len(z));
 for _ in range(5):
  p=np.linalg.lstsq(A*w[:,None],err*w,rcond=None)[0]; e=err-A@p; w=1/np.sqrt(1+(e/(2.5*m._mad(e)))**2)
 return p
for external in (False,True):
 results=[[] for _ in range(3)]
 for f in ([None] if external else range(5)):
  fit=np.ones(len(lab),bool) if external else fold!=f; v=~fit; sub={k:vv[fit] for k,vv in tr.items()}; val=te if external else {k:vv[v] for k,vv in tr.items()}; xv=xt if external else x[v]; cv=ct if external else co[v]; lv=val['lab_id']; ts,w=m._fit(sub,x[fit],co[fit],mp,dt,t,sy); p,b=pred(sub,x[fit],ts,w); err=slope(p,sub['strain_curves_obs']); obs=sub['instant_rate_obs']-b; drift=x[fit,1:]@w[3:7]
  coeffs=[]
  for l in range(4):
   q=(lab[fit]==l)&co[fit]; coeffs.append([correction(drift[q],y[q]) for y in (err,obs)])
  coeffs=np.array(coeffs); p,b=pred(val,xv,ts,w); z=xv[:,1:]@w[3:7]; inn=val['instant_rate_obs']-b; target=g['te_true_rate0']-b if external else slope(p,val['strain_curves_obs']); base=np.where(cv,.25,1.)*inn
  for j,kind in enumerate(('baseline','curve','curve+rate')):
   cf=coeffs[:,0] if j==1 else coeffs.mean(axis=1); shift=0 if j==0 else (cf[lv,0]+cf[lv,1]*z)*cv*.75
   pred_delta=base+shift; e=pred_delta-target; curve_err=p+pred_delta[:,None]*t-g['te_clean'] if external else e[:,None]*t
   results[j].append(np.column_stack((lv,cv,e,np.mean(curve_err**2,axis=1))))
  if external: print('Lab intercept and drift multiplier adjustments from training curve and onset:\n',coeffs.round(5))
 for kind,runs in zip(('baseline','curve','curve+rate'),results):
  a=np.concatenate(runs); mask=a[:,1].astype(bool); print('public' if external else 'OOF',kind,'complete rate/slope RMSE per lab',[round(np.sqrt(np.mean(np.clip(a[(a[:,0]==l)&mask,2],-.05,.05)**2)),6) for l in range(4)],'all curve RMSE per lab',[round(np.sqrt(np.mean(a[a[:,0]==l,3])),6) for l in range(4)] if external else '')
