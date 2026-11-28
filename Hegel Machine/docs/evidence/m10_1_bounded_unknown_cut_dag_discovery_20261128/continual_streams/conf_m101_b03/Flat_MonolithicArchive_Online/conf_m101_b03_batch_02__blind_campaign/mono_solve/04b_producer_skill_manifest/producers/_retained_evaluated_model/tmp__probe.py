import pickle, numpy as np
from scipy.optimize import least_squares, minimize_scalar
from sklearn.linear_model import ElasticNet, Ridge
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=S['train']; t=S['eval_times']
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(a): return np.c_[np.ones(len(a)),a]
def power(a,d): return np.column_stack([np.prod(a[:,k],axis=1) for j in range(1,d+1) for k in combinations_with_replacement(range(a.shape[1]),j)])
def coords(x):
 p=S['panel_a']; u,v=p['u'],p['v']; c=np.stack([fit(aff(u[:,i:i+1]),v[:,i]) for i in range(4)],axis=1)
 for i in range(25):
  e=np.linalg.norm(c[0]+c[1]*u-v,axis=1); w=np.sqrt(np.minimum(1,0.03/np.maximum(e,1e-12))); c=np.stack([fit(aff(u[:,j:j+1])*w[:,None],v[:,j]*w) for j in range(4)],axis=1)
 k=e<.01; c=np.stack([fit(aff(u[k,j:j+1]),v[k,j]) for j in range(4)],axis=1); z=(x-c[0])/c[1]
 p=S['panel_b']; b=aff(z)@fit(aff(p['u']),p['v']); out=[z,b];p=S['panel_c'];u=p['u']
 for j in (1,2):
  v=p['v'+str(j)]; best=(np.inf,None)
  for fun in (np.tanh,np.sin,np.cos):
   def d(x,f):return np.c_[aff(x),fun(f*x)]
   def loss(f):
    a=d(u,f);return np.linalg.norm(a@fit(a,v)-v)
   fs=np.linspace(.25,6,45); ii=np.argmin([loss(f) for f in fs]); f=minimize_scalar(loss,bounds=(fs[max(ii-1,0)],fs[min(ii+1,len(fs)-1)]),method='bounded').x
   print('C',j,fun.__name__,round(f,4),round(loss(f)/np.sqrt(v.size),6))
   if loss(f)<best[0]:best=(loss(f),d(z,f)@fit(d(u,f),v))
  out.append(best[1]); print('C polynomial LOOCV',[(d,np.mean(abs((v-(a:=aff(power(u,d)))@fit(a,v))/(1-np.diag(a@np.linalg.pinv(a)))[:,None]))) for d in (1,2)])
 return out

def modes(a):
 def basis(r): return -np.expm1(-t[:,None]*r)
 def r(p):
  d=basis(p);return (d@fit(d,a.T)-a.T).ravel()
 q=least_squares(r,[.35,1.6],bounds=([.04,.65],[.65,6])).x; d=basis(q);return d,fit(d,a.T).T,q

def cv(f,target,est='ridge',alpha=.01):
 n=52; a=f[:n]; m=a.mean(0); sc=np.maximum(a.std(0),1e-10); a=(a-m)/sc;b=(f[n:]-m)/sc
 if est=='ridge': return Ridge(alpha=alpha).fit(a,target[:n]).predict(b)
 return np.column_stack([ElasticNet(alpha=alpha,l1_ratio=.85,max_iter=20000,tol=1e-5).fit(a,v).predict(b) for v in target[:n].T])
if __name__=='__main__':
 z,b,c,w=coords(T['x']); d,a,q=modes(T['traj'][:52]); a=fit(d,T['traj'].T).T; y=np.c_[T['y'],a]; n=52
 print('rates',q,'in sample state projection',np.mean(abs(a@d.T-T['traj'])))
 for name,v in [('z',z),('b',b),('c',c),('w',w),('zc',np.c_[z,c]),('zw',np.c_[z,w]),('zcw',np.c_[z,c,w])]:
  for k in range(1,5 if v.shape[1]==4 else 3):
   f=power(v,k)
   for model,al in [('ridge',1e-4),('ridge',.1),('elastic',.001)]:
    p=cv(f,y,model,al);print(name,k,model,al,round(np.mean(abs(p[:,0]-y[n:,0])),6),round(np.mean(abs(p[:,1:]@d.T-T['traj'][n:])),6))
 for name,f in [('trig_b',np.c_[power(b,2),np.sin(b),np.cos(b),np.tanh(b)]),('mixed',np.c_[power(z,2),np.sin(b),np.cos(b),np.tanh(b),c,w]),('oracle_mode',power(np.c_[z,a],2))]:
  p=cv(f,y);print(name,np.mean(abs(p[:,0]-y[n:,0])),np.mean(abs(p[:,1:]@d.T-T['traj'][n:])))
