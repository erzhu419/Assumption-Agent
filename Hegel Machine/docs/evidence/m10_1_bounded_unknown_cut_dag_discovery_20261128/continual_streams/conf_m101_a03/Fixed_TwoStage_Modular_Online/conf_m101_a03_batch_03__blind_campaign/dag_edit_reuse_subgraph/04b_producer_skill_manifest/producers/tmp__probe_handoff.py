import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('sink','/holdout_task/_frozen_nodes/node_01.py'); sink=importlib.util.module_from_spec(spec); spec.loader.exec_module(sink)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=s['panel_a']; u,v=a['u'],a['v']; aff=lambda x:np.c_[np.ones(len(x)),x]; fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]; A=aff(u); ca=fit(A,v)
for _ in range(12):
 r=np.linalg.norm(v-A@ca,axis=1); w=1/np.hypot(1,r/max(np.median(r),1e-12)); ca=fit(A*w[:,None],v*w[:,None])
r=np.linalg.norm(v-A@ca,axis=1); keep=np.argsort(r)[:-3]; ca=fit(A[keep],v[keep]); ai=lambda x:np.linalg.solve(ca[1:].T,(x-ca[0]).T).T
b=s['panel_b']; u,v=b['u'],b['v']; x=ai(s['train']['x']); y=s['train']['y']; tr=s['train']['traj']
configs={
 'shared':([.4,2.85],lambda x,p:x-p[0]*np.sin(p[1]*x)),
 'separate':([.4]*4+[2.85]*4,lambda x,p:x-p[:4]*np.sin(p[4:]*x)),
 'shared+cubic':([.4,2.85,0],lambda x,p:x-p[0]*np.sin(p[1]*x)+p[2]*x**3),
 'shared+harmonic':([.4,2.85,0],lambda x,p:x-p[0]*np.sin(p[1]*x)+p[2]*np.sin(2*p[1]*x)),
 'shared+cross':([.4,2.85,0],lambda x,p:x-p[0]*np.sin(p[1]*x)+p[2]*x*np.sum(x*x,axis=1,keepdims=True)),
 'untied':([2.85]*4,lambda x,p:np.c_[x,np.sin(x*p)]),
}
folds=[np.arange(52,72),np.arange(0,20),np.arange(26,46)]
def metrics(z):
 vals=[]
 for te in folds:
  it=np.setdiff1d(np.arange(len(x)),te); data={'z':z[it],'z_test':z[te],'y':y[it],'traj':tr[it],'eval_times':s['eval_times'],'train_y':y[it],'train_traj':tr[it]};
  try: out=sink.run_node({'node_00':data}); vals.append([np.mean(abs(out['y_hat']-y[te])),np.mean(abs(out['state_pred']-tr[te]))]);
  except KeyError as e: print('missing',e); return
 return np.round(vals,6)
for name,(start,g) in configs.items():
 def solve(idx):
  def residual(p):
   d=aff(g(u[idx],p)); return (d@fit(d,v[idx])-v[idx]).ravel()
  p=least_squares(residual,start,max_nfev=100).x; return p,fit(aff(g(u[idx],p)),v[idx])
 idx=np.arange(len(u)); p,c=solve(idx); errs=[]
 for i in idx:
  pp,cc=solve(idx[idx!=i]); errs.append((aff(g(u[i:i+1],pp))@cc-v[i]).ravel())
 print(name,'parameters',p.round(6),'RMSE',np.sqrt(np.mean((aff(g(u,p))@c-v)**2)),'panel LOO',np.mean(np.abs(errs)), 'forward CV',metrics(aff(g(x,p))@c))
 print('unmixed at A-inverse',metrics(g(x,p)) if name!='untied' else '')
