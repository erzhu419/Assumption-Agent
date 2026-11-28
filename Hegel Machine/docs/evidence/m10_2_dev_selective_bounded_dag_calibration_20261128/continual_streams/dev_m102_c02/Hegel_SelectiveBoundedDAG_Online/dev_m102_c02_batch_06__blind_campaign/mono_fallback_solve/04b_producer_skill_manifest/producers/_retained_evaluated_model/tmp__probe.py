import numpy as np,pickle
from itertools import combinations_with_replacement as C
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=145)
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for i in range(1,d+1) for j in C(range(x.shape[1]),i)])
def fit(X,y):return np.linalg.lstsq(X,y,rcond=1e-10)[0]
a=s['panel_a']; u,v=a['u'],a['v']; good=np.linalg.norm(v,axis=1)<5
for l,r in [(u,v),(v,u)]:
 for d in [1,2,3]:
  X=poly(l[good],d); M=fit(X,r[good]); print('A',good.sum(),d,np.std(X@M-r[good],axis=0)); print(M if d==1 else '')
for k in ['panel_b','panel_c']:
 p=s[k];u=p['u'];
 for j,y in p.items():
  if j=='u':continue
  for name,X in [('lin',poly(u,1)),('quadr',poly(u,2)),('odd3',np.column_stack((poly(u,1),poly(u,3)[:,15:]))),('lin+sq',np.c_[poly(u,1),u*u]),('lin+cube',np.c_[poly(u,1),u**3])]:
   M=fit(X,y); print(k,j,name,'res',np.std(X@M-y,axis=0)); print(M if name in ['lin','lin+sq'] else '')
x,y,tr=[s['train'][k] for k in ('x','y','traj')]; z=np.c_[y,tr];print('Y from T',np.std(poly(tr,1)@fit(poly(tr,1)[:52],y[:52])-y)); print('corr',np.corrcoef(np.c_[x,z].T)[0:5]);
for d in [1,2,3,4]:
 X=poly(x,d);M=fit(X[:52],z[:52]);r=X@M-z;print('raw',d,np.mean(abs(r[:52]),0),np.mean(abs(r[52:]),0));
print('train,test means cov',x.mean(0),s['test']['x'].mean(0),np.cov(x.T))
