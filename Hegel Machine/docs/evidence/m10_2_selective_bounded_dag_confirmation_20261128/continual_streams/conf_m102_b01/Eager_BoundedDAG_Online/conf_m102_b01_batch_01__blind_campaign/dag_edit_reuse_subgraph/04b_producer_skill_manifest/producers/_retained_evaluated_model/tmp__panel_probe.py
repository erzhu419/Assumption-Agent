import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:len(y)]
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,z):return np.linalg.lstsq(x,z,rcond=None)[0]
def poly(x,d):return np.column_stack([np.prod(x[:,k],axis=1) for j in range(d+1) for k in combinations_with_replacement(range(x.shape[1]),j)])
def errs(a):
 z=np.c_[y,T]; r=a@fit(a,z)-z; p=a[52:]@fit(a[:52],z[:52])-z[52:]; return np.round([abs(r[:,0]).mean(),abs(r[:,1:]).mean(),abs(p[:,0]).mean(),abs(p[:,1:]).mean()],5)
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
b=s['panel_b']; c=s['panel_c']; bx,bv=b['u'],b['v']; cx,cv,dv=c['u'],c['v1'],c['v2']; B=fit(aff(bx),bv)
funcs={'atan':np.arctan,'asinh':np.arcsinh,'tanh':np.tanh,'inv3':lambda a:np.sinh(np.arcsinh(a)/3),'sin':np.sin,'cubic':lambda a:a+a**3,'softsign':lambda a:a/np.sqrt(1+a*a)}
def score(a,v):return np.mean((a@fit(a,v)-v)**2)**.5
def test(x,v):
 print('poly', [round(score(poly(x,k),v),5) for k in (1,2,3)])
 for k,f in funcs.items():
  q=minimize_scalar(lambda g:score(aff(f(x*g)),v),bounds=(.001,10),method='bounded'); print(k,round(q.x,4),round(q.fun,6),end='; ')
 print()
 for k,f in [('atan',np.tan),('asinh',np.sinh),('tanh',np.arctanh),('sin',np.arcsin),('softsign',lambda a:a/np.sqrt(1-a*a))]:
  mx=np.max(abs(v)); q=minimize_scalar(lambda g:score(aff(x),f(v*g))/g,bounds=(.001,min(8,.999/mx) if k in ('atan','tanh','sin','softsign') else 8),method='bounded'); print('out',k,round(q.x,4),round(q.fun,6),end='; ')
 print()
for nam,x,v in [('B',bx,bv),('C1',cx,cv),('C2',cx,dv),('C1->C2',cv,dv)]:
 print(nam);test(x,v)
for k,f in funcs.items():
 def vp(a):
  u=f(cx*np.exp(a[:4])+a[4:]); d=aff(u); return (d@fit(d,cv)-cv).ravel()
 q=least_squares(vp,np.r_[np.full(4,.7),np.zeros(4)],bounds=(-5,5),max_nfev=200); print('C1 shifted',k,np.exp(q.x[:4]),q.x[4:],np.std(q.fun))
for x,v in [(bx,bv),(cx,cv),(cx,dv),(cv,dv)]:
 print('lin residual sv',np.linalg.svd(aff(x)@fit(aff(x),v)-v,compute_uv=False),'output std',np.std(v,axis=0))
 print('mean approx inv-atan corr',np.corrcoef(np.c_[x, v].T)[:4,4:])
print('y traj',np.corrcoef(np.c_[y,T].T)[0]); U,S,V=np.linalg.svd(T,full_matrices=False); print('V',V[:2]); print('linear C1 projected',errs(aff(np.arctan(2.1*np.linalg.solve(B[1:].T,(w-B[0]).T).T))))
