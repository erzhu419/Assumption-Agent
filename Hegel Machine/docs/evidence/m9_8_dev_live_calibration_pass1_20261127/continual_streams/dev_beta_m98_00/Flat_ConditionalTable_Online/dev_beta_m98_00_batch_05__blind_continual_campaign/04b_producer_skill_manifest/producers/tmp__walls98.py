import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m98_00_batch_04_solver as b, dev_beta_m98_00_batch_03_solver as d
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=s['Z_nuisance']; src=s['source_modes']; y=d._calibration(s['sensor_calibration_sweeps'])(s['y_obs_raw']); cap=s['u_box']; total=s['total_budget']; ij=s['sensor_ij']; n=len(y)
# Estimate physical wall parameters using the observation equation, not the certificate's regularized objective.
for label,groups in [('nominal',[]),('shared',[[0,1,4,5]]),('axis',[[0,1],[4,5]]),('side',[[0],[1],[4],[5]]),('face',[[0,1],[4,5],[2,3],[6,7]]),('all',[[i] for i in range(8)])]:
 def calc(q):
  wall=np.ones(8)
  for indices,a in zip(groups,q):wall[indices]=a
  return b._response(s,wall,src)[0]
 def fit(train):
  zz=z[train]; proj=np.eye(sum(train))-zz@np.linalg.pinv(zz); yy=proj@y[train]
  def profile(q):
   S=calc(q)[train]; X=proj@S; w=b._weights(X,yy,1e-9,cap,total); gamma=np.linalg.lstsq(zz,y[train]-S@w,rcond=None)[0];return w,gamma,S@w+zz@gamma-y[train]
  q=least_squares(lambda q:profile(q)[2],np.ones(len(groups)),bounds=(0.05,5.),diff_step=1e-4,gtol=1e-11).x if groups else np.array([])
  w,gamma,r=profile(q); return q,w,gamma,np.sqrt(np.mean(r*r))
 q,w,g,r=fit(np.ones(n,bool));print(label,'wall',np.round(q,6),'w',np.round(w,6),'gamma',np.round(g,6),'in',round(r,7),flush=True)
 for fold in [np.repeat(np.arange(3),n//3), np.tile(np.repeat(np.arange(8)%4,4),3)]:
  out=[]
  for k in np.unique(fold):
   test=fold==k; a,v,h,_=fit(~test); r=(calc(a)@v+z@h-y)[test];out.append(np.sqrt(np.mean(r*r)))
  print('   held',np.round(out,7),flush=True)
