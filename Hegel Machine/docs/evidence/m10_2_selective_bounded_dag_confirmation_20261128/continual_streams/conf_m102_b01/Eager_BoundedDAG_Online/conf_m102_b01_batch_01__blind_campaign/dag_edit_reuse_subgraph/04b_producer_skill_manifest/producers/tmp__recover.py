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
from itertools import combinations
from scipy.optimize import least_squares
np.set_printoptions(precision=3,suppress=True)
b=s['panel_b']; c=s['panel_c']; bx,bv=b['u'],b['v']; cx,cv,dv=c['u'],c['v1'],c['v2']; B=fit(aff(bx),bv)
z=np.linalg.solve(B[1:].T,(w-B[0]).T).T

def lcv(a,v):
 u,ss,_=np.linalg.svd(a,full_matrices=False); h=np.sum(u[:,ss>1e-10]**2,axis=1); r=a@fit(a,v)-v
 return np.mean((r/np.maximum(1-h,1e-8)[:,None])**2)**.5
for name,x,v in [('B',bx,bv),('C2',cx,dv)]:
 print(name)
 mods=[('affine',lambda x:aff(x)),('square',lambda x:np.c_[aff(x),x*x]),('cubic',lambda x:np.c_[aff(x),x**3]),('cubic+square',lambda x:np.c_[aff(x),x*x,x**3]),('quadall',lambda x:poly(x,2)),('quadall+cubic',lambda x:np.c_[poly(x,2),x**3])]
 for label,f in mods:
  a=f(x);q=fit(a,v);print(label,np.std(a@q-v),lcv(a,v));
  if label in ('quadall','cubic','quadall+cubic'):print(q)
 for kind in ['cube','tanh','atan','asinh','sinh','square','rational']:
  def pred(p,X):
   A=p[:16].reshape(4,4); u=X@A+p[16:20]; g=p[20:24]; off=p[24:28]; f={'cube':lambda a:a**3,'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'sinh':lambda a:np.sinh(np.clip(a,-30,30)),'square':lambda a:a*a,'rational':lambda a:1/(1+a*a)}[kind]; return u+g*f(u)+off
  q=least_squares(lambda p:(pred(p,x)-v).ravel(),np.r_[fit(aff(x),v)[1:].ravel(),np.zeros(4),np.ones(4)*(.4 if kind in ['cube','sinh','square'] else -.4),np.zeros(4)],max_nfev=350);print('out',kind,np.std(q.fun),q.x[16:]);
  if name=='B' and np.std(q.fun)<.02:np.savez('/tmp/goodb_'+kind+'.npz',p=q.x,w=pred(q.x,z))
 print()
C1=lambda x:aff(np.arctan(2.11*x))@fit(aff(np.arctan(2.11*cx)),cv)
C2=lambda x:poly(x,2)@fit(poly(cx,2),dv)
ws={k:np.load('/tmp/goodb_'+k+'.npz')['w'] for k in ['cube','tanh','atan','asinh','sinh','square','rational'] if __import__('os').path.exists('/tmp/goodb_'+k+'.npz')}
for k,f in mods:ws[k]=f(z)@fit(f(bx),bv)
for k,u in list(ws.items())+[('raw',z)]:
 for label,x in [('B',u),('C1raw',C1(z)),('C1B',C1(u)),('C2raw',C2(z)),('C2B',C2(u)),('C1B/C2B',np.c_[C1(u),C2(u)]),('B/C2raw',np.c_[u,C2(z)]),('B/C1raw/C2raw',np.c_[u,C1(z),C2(z)])]:
  if x.shape[1]==4: es=[errs(aff(x)),errs(poly(x,2)),errs(poly(np.arctan(2*x),2)),errs(poly(np.arcsinh(3*x),2))]
  else:es=[errs(aff(x))]
  if min(max(a[2:]) for a in es)<.103 or k in ('raw','cubic','cube'):print(k,label,es)
