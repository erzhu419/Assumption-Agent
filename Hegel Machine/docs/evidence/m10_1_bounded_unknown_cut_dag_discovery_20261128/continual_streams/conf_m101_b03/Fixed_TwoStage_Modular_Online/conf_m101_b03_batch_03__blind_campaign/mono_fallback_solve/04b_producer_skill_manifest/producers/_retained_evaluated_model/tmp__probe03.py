import pickle,sys,importlib.util
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); t=s['eval_times']; T=tr['traj']; y=tr['y']; x=tr['x']; a=np.arange(52); b=np.arange(52,72)
def load(i):
 p=f'/tmp/hegel_continual_archive/conf_m101_b03_batch_0{i}_solver.py'; sp=importlib.util.spec_from_file_location('arc',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=load(1); c=m._calibrate(s['panel_a']); z=m._inverse(x,c); print('calibration',c.round(5)); X=np.c_[np.ones(n),z]; P=s['panel_b']; B=X@np.linalg.lstsq(np.c_[np.ones(len(P['u'])),P['u']],P['v'],rcond=None)[0];
for k in (2,3,4):
 def basis(r): return -np.expm1(-t[:,None]*r)
 def res(r):
  h=basis(r); return (h@np.linalg.lstsq(h,T[a].T,rcond=None)[0]-T[a].T).ravel()
 opt=least_squares(res,np.geomspace(.2,5,k),bounds=(.001,25),max_nfev=250); h=basis(opt.x); print('time',k,opt.x,np.sqrt(np.mean(res(opt.x)**2)),np.sqrt(np.mean((T[b]-T[b]@np.linalg.pinv(h.T)@h.T)**2)))
for i in range(3):
 try:
  q=s.copy();q['train']={k:v[:52] if k!='summary' else v for k,v in tr.items()};q['test']={'x':x[b]};out=load(i).solve_batch(q);print('archive',i,np.mean(abs(out['y_hat']-y[b])),np.mean(abs(out['state_pred']-T[b])))
 except Exception as e:print('archive error',i,e)
for name,F in [('raw',x),('latent',z),('B',B),('latent+B',np.c_[z,B])]:
 F=(F-F[a].mean(0))/F[a].std(0); D=cdist(F,F,'sqeuclidean')/F.shape[1]; Y=np.c_[y,T]; results=[]
 for label,K in [(f'p{d}',(1+F@F.T/F.shape[1])**d) for d in range(1,6)]+[(f'r{g}',np.exp(-g*D)) for g in (.05,.15,.4,1,2)]:
  for r in (1e-7,1e-4,.003,.03,.15):
   pred=Y[a].mean(0)+K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+r*np.eye(len(a)),Y[a]-Y[a].mean(0),assume_a='pos'); e=np.mean(abs(pred-Y[b]),axis=0); results.append((e[0],e[1:].mean(),label,r))
 print(name,'y',sorted(results)[:3],'T',sorted(results,key=lambda p:p[1])[:3])
