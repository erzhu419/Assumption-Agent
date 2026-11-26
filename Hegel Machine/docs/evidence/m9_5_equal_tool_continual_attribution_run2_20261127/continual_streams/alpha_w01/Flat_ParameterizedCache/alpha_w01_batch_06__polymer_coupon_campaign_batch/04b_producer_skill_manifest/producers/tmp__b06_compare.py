import importlib.util,pickle,numpy as np

def load(path):
 spec=importlib.util.spec_from_file_location('_m',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

root='/holdout_task/'
d=pickle.load(open(root+'public_alpha_w01_batch_06.pkl','rb'))[0]['dataset']; tr=d['train']; dt,t,sy=(d[k] for k in ('past_dt','eval_times','yield_stress'))
lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(1)
for f in ('/tmp/b06_predecessor.py',root+'alpha_w01_batch_06_solver.py'):
 m=load(f); maps=m._calibrate(d['calibration_load_cells']); h,u=m._stress(tr,maps)
 y=tr['strain_curves_obs']; r=tr['instant_rate_obs']; E=np.empty_like(y); R=np.empty_like(r); D=np.empty_like(r); T=[]
 for l in range(4):
  take=lab==l; sub=lambda i:{k:v[i] for k,v in tr.items()}; a,b=sub(~take),sub(take); xa,xb,ca,cb=m._align(a,b)
  tau,theta=m._fit(a,xa,ca,maps,dt,t,sy); T.append(tau)
  ov,w=m._plastic(h[take],u[take],dt,sy); A,B=m._design(tau,h[take],u[take],xb,dt,t,ov,w)
  D[take]=r[take]-B@theta; correction=np.where(cb,0.,D[take]); R[take]=correction-D[take]; E[take]=b['strain_0'][:,None]+A@theta+correction[:,None]*t-y[take]
 print(f, 'maps:',maps.tolist(),'LOLO taus:',np.round(T,5).tolist())
 rmse=lambda a: np.sqrt(np.mean(a*a)); maskout=abs(u)>sy
 for l in range(4):
  for state in (False,True):
   i=(lab==l)&(maskout==state); print('lab/above/n',l,state,i.sum(),'curve/rate/worst horizon',np.round([rmse(E[i]),rmse(R[i]),max(np.sqrt(np.mean(E[i]**2,axis=0)))],6))
 i=~complete; print('missing curve RMSE: full/half/no onset correction',np.round([rmse(E[i]-a*D[i,None]*t) for a in (0,.5,1)],6))
 print('complete curve RMSE: no/quarter/full onset correction',np.round([rmse(E[complete]+a*D[complete,None]*t) for a in (0,.25,1)],6))
