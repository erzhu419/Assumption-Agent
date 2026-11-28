import numpy as np
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist

def aff(u): return np.c_[np.ones(len(u)),u]
def quad(u): return np.column_stack([aff(u)]+[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(u.shape[1]),2)])
def maps(s,xx,debug=False):
 a,b,c=(s['panel_'+j] for j in 'abc'); u=a['u']; v=a['v']; good=np.max(abs(u-v),1)<1
 m=np.array([np.linalg.lstsq(aff(u[good][:,j:j+1]),v[good,j],rcond=None)[0] for j in range(4)]); z=(xx-m[:,0])/m[:,1]
 mb=np.linalg.lstsq(aff(b['u']),b['v'],rcond=None)[0]
 u=c['u']; v=c['v1']; w=c['v2']; h=quad(u); p1=np.linalg.lstsq(h,v,rcond=None)[0]
 def v2f(u,r,lin):
  q=np.expm1(-(u-r[4:])**2*np.exp(r[:4])); return np.c_[aff(u),q] if lin else aff(q)
 fits=[]
 for lin in (False,True):
  for k in (.3,1.,3.):
   def fun(r):
    h=v2f(u,r,lin); return (h@np.linalg.lstsq(h,w,rcond=None)[0]-w).ravel()
   opt=least_squares(fun,np.r_[np.full(4,np.log(k)),np.zeros(4)],bounds=([-5]*4+[-1.5]*4,[3]*4+[1.5]*4),max_nfev=220); fits.append((np.std(opt.fun),lin,opt.x))
 er,lin,r=min(fits,key=lambda a:a[0]); h2=v2f(u,r,lin); p2=np.linalg.lstsq(h2,w,rcond=None)[0]
 if debug: print('B',mb,'C1 residual',np.std(h@p1-v),'C1 params',p1,'C2 residual/params',er,lin,r,p2,flush=True)
 return z,aff(z)@mb,quad(z)@p1,v2f(z,r,lin)@p2,mb,p1,p2,r,lin

def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sparse = inputs['node_00']
    tr = study['train']
    n = len(tr['x'])
    _, _, c, d, *_ = maps(study, np.vstack((tr['x'], study['test']['x'])))
    f = np.c_[c,d]
    f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-9)
    k = (1 + f @ f[:n].T/f.shape[1])**2
    a = k[:n]; b = k[n:]; m = a.mean(0)
    a = a-m[:,None]-m+m.mean()
    b = b-b.mean(1)[:,None]-m+m.mean()
    y = np.c_[tr['y'],tr['traj']]; ym = y.mean(0)
    p = b @ solve(a+.001*np.eye(n),y-ym,assume_a='pos')+ym
    t = study['eval_times']; q = -np.expm1(-t[:,None]*[.425,1.832]); p[:,1:] = (p[:,1:] @ np.linalg.pinv(q).T) @ q.T
    p[:,0] = 2*p[:,0]-sparse['y_hat']
    p[:,1:] = (p[:,1:]-.35*sparse['state_pred'])/.65
    return {'panel_pred':p}
