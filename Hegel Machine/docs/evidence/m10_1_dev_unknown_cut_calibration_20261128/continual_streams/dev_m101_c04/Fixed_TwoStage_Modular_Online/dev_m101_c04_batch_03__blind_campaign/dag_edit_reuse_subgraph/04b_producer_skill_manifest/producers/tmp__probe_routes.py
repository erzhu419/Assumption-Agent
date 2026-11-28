import pickle, numpy as np, runpy
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.column_stack((np.ones(len(x)),x))
def quad(x):return np.column_stack([aff(x)]+[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)])
u,v=(s['panel_a'][k] for k in ('u','v')); D=aff(u); w=ls(D,v)
for _ in range(15):
 wt=np.maximum(np.linalg.norm(D@w-v,axis=1),1e-8)**-0.5; w=ls(D*wt[:,None],v*wt[:,None])
keep=np.argsort(np.linalg.norm(D@w-v,axis=1))[:-3];w=ls(D[keep],v[keep]); A=lambda x:np.linalg.solve(w[1:].T,(x-w[0]).T).T
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; ax=A(x)
def scores(z):
 vals=[]
 for i in (0,20,40,52):
  te=np.arange(i,i+20); tr=np.setdiff1d(np.arange(len(z)),te)
  r=sink({'node_00':dict(z_tr=z[tr],z_te=z[te],y=y[tr],traj=t[tr],eval_times=s['eval_times'])}); vals.append([np.mean(abs(y[te]-r['y_hat'])),np.mean(abs(t[te]-r['state_pred']))])
 return np.asarray(vals).round(5).tolist()
print('raw',scores(x),'A inverse',scores(ax)); print('A x quantiles',np.quantile(ax,[0,.1,.5,.9,1],axis=0)); print('trajectory latent sv',np.linalg.svd(t-t.mean(0),compute_uv=False))
for pn,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 a,b=s[pn]['u'],s[pn][k]; print(pn,k,'u q',np.quantile(a,[0,.1,.5,.9,1],axis=0).round(3));print('v q',np.quantile(b,[0,.1,.5,.9,1],axis=0).round(3));
 for lab,uu,vv in [('f',a,b),('r',b,a)]:
  for f in (aff,quad):
   W=ls(f(uu),vv); print(lab,f.__name__,'from A',scores(f(ax)@W))
 d=aff(a); e=b-d@ls(d,b); print('residual sv',np.linalg.svd(e,compute_uv=False).round(7));print('quadratic trajectory residual from A',np.mean(abs(t-quad(ax)@ls(quad(ax),t))))
