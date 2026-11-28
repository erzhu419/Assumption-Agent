import sys,pickle,importlib.util,numpy as np
from scipy.optimize import least_squares
root='/tmp/hegel_continual_archive/'
def load(i):
 s=importlib.util.spec_from_file_location('solver',root+f'dev_alpha_m100_01_batch_{i:02d}_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(7); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr,te,d=ds['train'],ds['test'],item['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; n=len(tr['lab_id']); x=m._features(tr,te); xd=m._features(tr,d)[n:]; b=m._bridge(ds['calibration_load_cells']);  p,h=m._loads(tr,b)
def obs(q):return q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']
c,r=obs(tr); T,w=m._fit(p,h,x[:n],dt,t,c,r); print('TAUS',T,'THETA',w,'bridge',b.round(5));
def stats(label,curve,rate,c,r):
 e=curve-c; f=rate-r; print(label,'rmse',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(f**2)),'median',np.median(abs(e)),np.median(abs(f)),'curve by time',np.sqrt(np.mean(e**2,axis=0)).round(4))
A,B=m._design(T,p,h,x[:n],dt,t);stats('train',A@w,B@w,c,r)
p,h=m._loads(d,b); c,r=obs(d); A,B=m._design(T,p,h,xd,dt,t);stats('baseline',A@w,B@w,c,r); stats('07',*m._chamber_response(T,w,p,h,xd,dt,t),c,r)
for i in [1,2,3,4,5,6]:
 j=load(i); fun=getattr(j,'_chamber_response',None)
 if fun:
  args={'taus':T,'theta':w,'past':p,'hold':h,'x':xd,'dt':dt,'t':t}; import inspect
  try: stats(str(i),*fun(*(args[k] for k in inspect.signature(fun).parameters)),c,r)
  except KeyError as e: print(e)
R=-np.expm1(-t[:,None]/T); D=w[2]+xd@w[3:]; z=h[:,None]-B[:,:2]*T
coef=np.linalg.lstsq(np.vstack((np.column_stack((R,t)),np.r_[1/T,1.])),np.column_stack((c,r)).T,rcond=None)[0].T
F=np.column_stack((np.ones(len(h)),h,z,D,p,xd));print('coef ranges',coef.min(0),coef.max(0));print('correlation columns 1 hold memories drift past x',np.corrcoef(np.column_stack((coef,F[:,1:])).T)[:3,3:].round(3))
for k,ids in [('hold+drift',[0,1,4]),('memory+hold+drift',[0,1,2,3,4]),('past+hold+drift',[0,1,4,*range(5,11)])]:
 f=F[:,ids]; a=np.linalg.lstsq(f,coef,rcond=None)[0]; pred=f@a; print(k,a.T.round(5));stats(k,pred@R.T if False else pred@np.column_stack((R,t)).T,pred@np.r_[1/T,1.],c,r)
