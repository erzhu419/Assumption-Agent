import sys,pickle,importlib.util,itertools
import numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; a,b=c['ref_rosette'],c['raw_rosette']; clean=np.max(abs(b),axis=1)<5
for fold in range(4):
 sel=np.arange(len(a))%4==fold; f,p,F=m._sensor_fit(s,~sel); e=F(a[sel&clean])-b[sel&clean]; print('cal',fold,f,p.round(5) if fold==0 else '',np.sqrt(np.mean(e*e,axis=0)))
y=m._calibrate(s,s['y_obs_raw']); z=s['Z_nuisance']; r=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; ridge=s['nuisance_ridge']; n=len(y)
def compare(S):
 def solve(t):
  A=np.c_[S[t],y[t]]; Z=z[t]; p=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T); B=A-Z@(p@A); D,b=B[:,:-1],B[:,-1]; w,nu=m._box_qp(D.T@D+r*np.eye(D.shape[1]),-D.T@b,cap,budget); return w,p@(y[t]-S[t]@w),D@w-b
 w,g,res=solve(np.ones(n,bool)); cv=[]
 groups=[np.tile(np.arange(n//12),12),np.tile(np.repeat(np.arange(n//12),4),3),np.repeat(np.arange(3),n//3),np.tile(np.arange(n//3),3)]
 for group in groups[1:]:
  out=np.empty(n)
  for j in np.unique(group):
   sel=group==j; ww,gg,_=solve(~sel); out[sel]=S[sel]@ww+z[sel]@gg-y[sel]
  cv.append((np.sqrt(np.mean(out**2)),max(np.sqrt(np.mean(out[group==j]**2)) for j in np.unique(group))))
 return np.sqrt(np.mean(res**2)),cv,w
for wall in itertools.product([0,1],repeat=2):
 S,U=m._responses(s,wall); print('wall',wall,compare(S))
p='/tmp/hegel_continual_archive/beta_m100_c03_batch_02_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old); print('archive02',compare(old._responses(s)[0]));
