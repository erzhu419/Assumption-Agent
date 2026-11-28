import numpy as np,pickle,importlib.util,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; t=s['eval_times']; T=tr['traj']; n=len(x)
sp=importlib.util.spec_from_file_location('base','/holdout_task/_frozen_nodes/node_00.py'); base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
def timeq(r):return -np.expm1(-t[:,None]*r)
def err(r):
 q=timeq(r); return (T-np.linalg.lstsq(q,T.T,rcond=None)[0].T@q.T).ravel()
r=least_squares(err,[.392,1.857],bounds=(.02,8)).x; q=timeq(r); a=np.linalg.lstsq(q,T.T,rcond=None)[0].T; Y=np.c_[y,a]; print('rates',r,'noise',np.std(err(r)),'target std',Y.std(0));
u=s['panel_a']['u'];v=s['panel_a']['v']; h=np.c_[np.ones(len(u)),u]; keep=np.linalg.norm(v-h@np.linalg.lstsq(h,v,rcond=None)[0],axis=1)<3;u=u[keep];v=v[keep]; h=h[keep]; c=np.linalg.lstsq(h,v,rcond=None)[0];print('panel A clean linear', np.std(v-h@c,0),'coef',c)
for name,f in [('quad',u**2),('tanh',np.tanh(u)),('sin',np.sin(u)),('sat',np.expm1(-.2962*u*u)),('cube',u**3)]:
 d=np.c_[h,f]; m=np.linalg.lstsq(d,v,rcond=None)[0]; print(name,np.std(v-d@m,0),m[5:].round(3))
C=np.diag(np.diag(c[1:])); zsets={'raw':x,'diag':(x-c[0])@np.linalg.inv(C),'full':(x-c[0])@np.linalg.inv(c[1:])}; zsets['cyclic']=(x-c[0])@np.linalg.inv(c[1:]*(np.eye(4)+np.roll(np.eye(4),-1,axis=0)))
inds=np.arange(n); spidx=52; study=dict(s,train={k:tr[k][:spidx] for k in ['x','y','traj']},test={'x':x[spidx:]}); b=base.run_node({'__input__':study}); print('base MAE',np.mean(abs(b['y_hat']-y[spidx:])),np.mean(abs(b['state_pred']-T[spidx:])))
for nm,z in zsets.items():
 print(nm,'cov',np.corrcoef(z.T).round(2))
 for k in (1,2,3,4):
  f=np.column_stack([np.prod(z[:,j],axis=1) for p in range(1,k+1) for j in it.combinations_with_replacement(range(4),p)]);f=(f-f[:52].mean(0))/np.maximum(f[:52].std(0),1e-8)
  for al in ([.001] if k==1 else [.001,.1,1.]):
   p=f[52:]@f[:52].T@np.linalg.solve(f[:52]@f[:52].T+al*np.eye(52),Y[:52]-Y[:52].mean(0))+Y[:52].mean(0);print(k,al, np.mean(abs(p-Y[52:]),0).round(4),round(np.mean(abs(p[:,1:]@q.T-T[52:])),4))
