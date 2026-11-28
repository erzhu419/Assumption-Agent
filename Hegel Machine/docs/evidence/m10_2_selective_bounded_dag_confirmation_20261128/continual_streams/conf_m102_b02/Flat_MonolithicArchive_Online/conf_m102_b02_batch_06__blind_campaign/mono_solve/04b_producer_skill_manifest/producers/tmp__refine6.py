import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import _coords,_i,_ls,_cross,_ridge,_response,_calibrate,_pair,_operator
import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];a=s['train']; y=a['y'];tr=a['traj'];n=len(y);z=_calibrate(s['panel_a'],a['x']); b=s['panel_b']; c=s['panel_c']; u,v=b['u'],b['v']; rng=np.random.default_rng(229);folds=[np.arange(n),*[rng.permutation(n) for _ in range(8)]]
for name,fn,p in [('sine+cub',lambda x,p:x-p[0]*np.sin(p[1]*x)+p[2]*x**3,[.4,2.8,0]),('sat+cub',lambda x,p:x-p[0]*x/(1+np.exp(p[1])*x*x)+p[2]*x**3,[1,-.2,0]),('sine+cos',lambda x,p:x-p[0]*np.sin(p[1]*x)+p[2]*(1-np.cos(p[1]*x)),[.4,2.8,0]),('odd5',lambda x,p:x+p[0]*x**3+p[1]*x**5,[.7,0]),('power',lambda x,p:x-p[0]*x/(1+np.exp(p[1])*x*x)**np.exp(p[2]),[1,-.2,0])]:
 def fit(p):
  D=_i(fn(u,p));return (D@_ls(D,v)-v).ravel()
 f=least_squares(fit,p,max_nfev=180); print('B',name,f.x,np.abs(f.fun).mean(),flush=True)
cc=lambda x,p:_i(x+p[0]*(1-np.cos(p[1]*x)));cp=[.72,1.2]; print('C cos fit',np.abs(_operator(c['u'],c['v2'],c['u'],cc,cp)-c['v2']).mean())
for bn in ['sat','sine']:
 z,w,q,r0=_coords(s,a['x'],bn,'c');r1=_operator(c['u'],c['v2'],w,cc,cp)
 for rn,r in [('cub',r0),('cos',r1)]:
  blocks=(z,w,q,r); cross=[_cross(p,q) for p,q in itertools.combinations(blocks,2)]; base=np.c_[z,q,r];
  for nm,D in [('all',np.c_[base,*cross]),('nozw',np.c_[base,*cross[1:]]),('zq_qr',np.c_[base,cross[1],cross[-1]]),('no_qr',np.c_[base,*cross[:-1]]),('outerzq',np.c_[base,(z[:,:,None]*q[:,None,:]).reshape(n,-1)]),('diag',np.c_[base,*[p*q for p,q in itertools.combinations(blocks,2)]])]:
   d=_i(D)
   for alpha in [.005,.03,.15]:
    for k in [1,3]:
     e=[]
     for f in folds:
      i,j=f[:52],f[52:];m=tr[i].mean(0);modes=np.linalg.svd(tr[i]-m,full_matrices=False)[2][:k]; t=(tr[i]-m)@modes.T;p=_ridge(d[i],t,d[j],alpha); pred=_response(_i(np.c_[z[i],q[i],t]),y[i],_i(np.c_[z[j],q[j],p]),np.tanh); e.append([abs(pred-y[j]).mean(),abs(p@modes+m-tr[j]).mean()]);
     e=np.array(e);print('CV',bn,rn,nm,alpha,k,*np.round(np.r_[e[0],e[1:].mean(0),e[1:].max(0)],5),flush=True)
