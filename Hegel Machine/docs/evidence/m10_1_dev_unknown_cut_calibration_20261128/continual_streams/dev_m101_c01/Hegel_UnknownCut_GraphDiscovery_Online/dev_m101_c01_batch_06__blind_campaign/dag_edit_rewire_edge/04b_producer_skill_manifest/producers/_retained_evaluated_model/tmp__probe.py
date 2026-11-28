import pickle, numpy as np, runpy
from itertools import combinations_with_replacement as cwr
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['y']); a=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); k=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']({'node_00':a}); x=a['x'][:n]; t=s['train']['traj']; y=s['train']['y'];
print('traj singular values',np.linalg.svd(t,compute_uv=False));print('x ranges',x.min(0),x.max(0))
def quad(x): return np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in cwr(range(4),2)]])
def err(f,y):
 w=np.linalg.pinv(f); r=y-f@(w@y); h=(f*w.T).sum(1); z=r/(1-h)[:,None]; return np.round(np.abs(r).mean(0),6),np.round(np.abs(z).mean(0),6)
for name,z,out in [('y',x,y[:,None]),('traj',x,t),('B',s['panel_b']['u'],s['panel_b']['v']),('C1',s['panel_c']['u'],s['panel_c']['v1']),('C2',s['panel_c']['u'],s['panel_c']['v2'])]:
 print(name,'range',out.min(0),out.max(0))
 for label,f in [('linear',lambda z:quad(z)[:,:5]),('quad',quad)]+[(f'{fn.__name__}{c}',lambda z,fn=fn,c=c:quad(fn(c*z))) for fn in (np.tanh,np.sin) for c in (.5,1,1.5,2)]: print(label,*err(f(z),out))
for name,ks in k.items():
 for j,K in enumerate(ks):
  K=K[:n,:n]; ss,U=np.linalg.eigh(K); best=[]
  for lam in np.logspace(-7,1,17):
   I=(U*(1/np.maximum(ss+lam,lam)))@U.T; r=(I@np.column_stack([y,t]))/I.diagonal()[:,None]; best.append((lam,np.abs(r[:,0]).mean(),np.abs(r[:,1:]).mean()))
  print(name,j,min(best,key=lambda a:a[1]),min(best,key=lambda a:a[2]))
