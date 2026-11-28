import pickle, numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(u):return np.c_[np.ones(len(u)),u]
def cal(p,full=False):
 u,v=p['u'],p['v']; k=np.ones(len(u),bool)
 for i in range(10):
  a,b=u[k],v[k]; ac=a-a.mean(0); bc=b-b.mean(0); m=ls(ac,bc) if full else np.diag(np.sum(ac*bc,0)/np.sum(ac**2,0)); o=b.mean(0)-a.mean(0)@m; r=np.linalg.norm(u@m+o-v,axis=1); k=r<max(5*np.median(r),.0006)
 return m,o,r
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True)
 for full in (False,True):
  m,o,r=cal(s['panel_a'],full); print('cal',m,o,np.sort(r)[::4])
 for key in ('v1','v2'):
  u=s['panel_c']['u']; v=s['panel_c'][key]; a=aff(u); i=np.arange(len(u)); mask=i%4!=0
  def report(f):
   return np.sqrt(np.mean((f@ls(f,v)-v)**2)),np.sqrt(np.mean((f[~mask]@ls(f[mask],v[mask])-v[~mask])**2))
  for name,f in [('aff',a),('cross',np.c_[a,*[u[:,i]*u[:,j] for i,j in combinations(range(4),2)]]),('quad',np.c_[a,*[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]])]:print(key,name,report(f))
  for kind in ('sin','cos','sin2','cos2','sc','tanh','rational'):
   def f(w):
    ft={'sin':[np.sin(w*u)],'cos':[np.cos(w*u)],'sin2':[np.sin(w*u),np.sin(2*w*u)],'cos2':[np.cos(w*u),np.cos(2*w*u)],'sc':[np.sin(w*u),np.cos(w*u)],'tanh':[np.tanh(w*u)],'rational':[u/(1+w*u*u)]}[kind]; return np.column_stack([a,*ft])
   def err(w):return report(f(w))[0]
   ws=np.arange(.15,10,.05); es=np.array([err(w) for w in ws]); cand=np.r_[False,(es[1:-1]<=es[:-2])&(es[1:-1]<=es[2:]),False]; idx=np.flatnonzero(cand); idx=np.unique(np.r_[idx,np.argmin(es)]); idx=idx[np.argsort(es[idx])][:3]; out=[]
   for j in idx:
    w=minimize_scalar(err,bounds=(max(.01,ws[j]-.05),ws[j]+.05),method='bounded').x; out.append((round(w,4),*np.round(report(f(w)),6)))
   print(key,kind,out)
 t=s['train']['traj']; print('trajectory singular values',np.linalg.svd(t,compute_uv=False)); print('trajectory spread',t.std(0),'y spread',s['train']['y'].std())
