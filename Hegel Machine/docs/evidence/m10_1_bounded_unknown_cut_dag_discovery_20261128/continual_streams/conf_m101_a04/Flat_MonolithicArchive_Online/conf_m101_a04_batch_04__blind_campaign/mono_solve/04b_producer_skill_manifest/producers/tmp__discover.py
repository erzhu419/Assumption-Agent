import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def bias(a): return np.c_[np.ones(len(a)),a]
def quad(a): return np.column_stack([a[:,i]*a[:,j] for i,j in cwr(range(a.shape[1]),2)])
def calibrate(p,x):
 u,v=p['u'],p['v']; k=np.ones(len(u),bool)
 for _ in range(8):
  a,b=u[k],v[k]; ac=a-a.mean(0); w=(ac*(b-b.mean(0))).sum(0)/(ac**2).sum(0); q=b.mean(0)-a.mean(0)*w; e=np.linalg.norm(u*w+q-v,axis=1); nk=e<max(.0006,5*np.median(e))
  if np.array_equal(k,nk): break
  k=nk
 return (x-q)/w
x=calibrate(s['panel_a'],s['train']['x']); u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']
f1=lambda a:bias(np.c_[a,quad(a)])
c1=fit(f1(u),v1); print('v1 coeff',np.round(c1,4).T); c=f1(x)@c1
bs=[('cos common',lambda a,p:np.c_[bias(a),np.cos(p[0]*a)],[1.25]),('cos separate',lambda a,p:np.c_[bias(a),np.cos(p*a)],[1.25]*4),('cos2 common',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),np.cos(p[1]*a)],[1.1,3.9]),('cos harmonic',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),np.cos(2*p[0]*a)],[1.25]),('cos sin',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),np.sin(p[1]*a)],[1.25,3]),('cos quad',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),quad(a)],[1.25])]
bp=s['panel_b']; b=bias(x)@fit(bias(bp['u']),bp['v']); T=np.c_[-np.expm1(-1.8283*t),t/(1+.41587*t)]; targets=np.c_[y,z,fit(T,z.T).T]
for name,f,p in bs:
 def residual(p):
  a=f(u,p);return (a@fit(a,v2)-v2).ravel()
 q=least_squares(residual,p,bounds=(.15,8),gtol=1e-12); d=f(x,q.x)@fit(f(u,q.x),v2); print(name,np.round(q.x,5),np.sqrt(np.mean(q.fun**2)))
 xy=[('cd',np.c_[c,d]),('cd-match',np.c_[c,d,(c*d).sum(1)]),('cd-pair',np.c_[c,d,c*d,c[:,0]*d[:,1]]),('c',c),('d',d),('cd-cross',np.c_[c,d,(c[:,:,None]*d[:,None,:]).reshape(-1,16)]),('cd-full',np.c_[c,d,quad(np.c_[c,d])]),('bcd',np.c_[b,c,d,(c*d).sum(1),c[:,0]*d[:,1]]),('xcd',np.c_[x,c,d,(c*d).sum(1),c[:,0]*d[:,1]])]
 for nm,a in xy:
  A=bias(a); err=np.abs(A[52:]@fit(A[:52],targets[:52])-targets[52:]).mean(0);print(nm,np.round([err[0],err[1:9].mean(),err[-2],err[-1]],5),end='; ')
 print()
