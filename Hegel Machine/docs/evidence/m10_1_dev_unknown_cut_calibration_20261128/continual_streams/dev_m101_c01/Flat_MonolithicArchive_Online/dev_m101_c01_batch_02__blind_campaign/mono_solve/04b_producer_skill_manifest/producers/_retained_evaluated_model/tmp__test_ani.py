import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as m
import numpy as np, pickle
from anisotropic import extrapolate
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; y=d['y']; z=d['traj']; n=len(x)
a,b=m.panel_coordinates(s,x); rng=np.random.default_rng(39217)
sets=[np.arange(56,72)]+[rng.permutation(n)[:16] for _ in range(16)]+[np.argsort(abs(x[:,j]))[-16:] for j in range(4)]+[np.argsort(np.linalg.norm(x-x[np.argmax(abs(x[:,j]))],axis=1))[:16] for j in range(4)]
res={}
for te in sets:
 tr=np.setdiff1d(np.arange(n),te); sub=dict(s,train={k:d[k][tr] for k in ('x','traj','y')},test={'x':x[te]}); old=m.solve_batch(sub); f=m.temporal_basis(s['eval_times'],z[tr]); u=np.linalg.lstsq(f,z[tr].T,rcond=None)[0].T
 def record(k,yp,zp):
  e=abs(zp-z[te]); res.setdefault(k,[]).append([np.mean(abs(yp-y[te])),e.mean(),e[:,-1].mean()])
 record('incumbent',old['y_hat'],old['state_pred'])
 for name,w in [('direct',a),('inverse',b),('raw',x)]:
  yp=extrapolate(w[tr],w[te],y[tr],m.quadratic); zp=extrapolate(w[tr],w[te],u,m.quadratic)@f.T
  for mix in (.35,.7,1.):
   record((name,mix),mix*yp+(1-mix)*old['y_hat'],mix*zp+(1-mix)*old['state_pred'])
for k,v in res.items():
 v=np.array(v); print(k, 'gate',v[0].round(5),'random mean/max',v[1:17].mean(0).round(5),v[1:17].max(0).round(5),'blocked mean/max',v[17:].mean(0).round(5),v[17:].max(0).round(5))
