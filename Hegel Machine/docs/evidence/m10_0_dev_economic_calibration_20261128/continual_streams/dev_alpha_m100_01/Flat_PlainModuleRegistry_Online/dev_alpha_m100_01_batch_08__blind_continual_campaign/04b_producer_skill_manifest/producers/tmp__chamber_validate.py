from chamber_data import *
from downstream_solver import _predict
l=di['lab_id']; z=h[:,None]-g; one=np.ones(len(h)); a,bs,d=B.T; base=c.copy();base[:,0]=th[0]*h; Y=y-base@B.T
# Compare chamber-memory contrasts and coupon-feature effects on held-out labs.
samples=np.c_[np.zeros(len(h)),p,h]; jumps=np.diff(samples,axis=1)
V={'z*s':z[:,1]*s,'x3*s':xx[:,3]*s,'z':z[:,1],'zfast':z[:,0],'zfast*s':z[:,0]*s,'abs(z)':abs(z[:,1]),'h-z':g[:,1],'h*s':h*s,'z^2':z[:,1]**2,'rise':np.maximum(jumps[:,:-1],0).sum(1),'fall':np.maximum(-jumps[:,:-1],0).sum(1),'lastRise':np.maximum(jumps[:,-2],0),'lastFall':np.maximum(-jumps[:,-2],0),'holdRise':np.maximum(jumps[:,-1],0),'holdFall':np.maximum(-jumps[:,-1],0),'peak':p.max(1),'trough':p.min(1)}
for j in range(4): V[f'x{j}']=xx[:,j]; V[f'x{j}*s']=xx[:,j]*s
models=[(),('z*s',)]+[(k,) for k in V if k!='z*s']+[('z*s',k) for k in V if k!='z*s']
def fit(ks,m):
 A=np.stack([s[:,None]*a,s[:,None]*d,one[:,None]*bs,s[:,None]*bs]+[V[k][:,None]*bs for k in ks],axis=-1)
 beta=np.linalg.lstsq(A[m].reshape(-1,A.shape[-1]),Y[m].ravel(),rcond=None)[0]; return Y-A@beta,beta
def metrics(r): return np.r_[np.sqrt(np.mean(r[:,:-1]**2)),np.sqrt(np.mean(r[:,-1]**2))]
def cv(ks):
 stats=np.array([metrics(fit(ks,l!=i)[0][l==i]) for i in np.unique(l)]); return stats.max(0),stats
for ks in [(),('z*s',),('z*s','x3*s')]: print('candidate',ks,'worst',cv(ks)[0],'by lab',cv(ks)[1].round(5),'full fit',fit(ks,one==1)[1])
for wor,k in sorted((tuple(cv(k)[0]),k) for k in models)[:10]: print('ranked',wor,k)
for family in (models, [(),('z*s',),('z*s','x3*s')]):
 e=np.empty_like(Y)
 for i in np.unique(l):
  train=l!=i
  def inner(ks):
   rs=[metrics(fit(ks,train&(l!=j))[0][l==j]) for j in np.unique(l[train])]; return np.mean(rs)
  k=min(family,key=inner); e[~train]=fit(k,train)[0][~train]; print('inner selection',i,k)
 print('nested pooled',metrics(e),'nested worst',np.max([metrics(e[l==i]) for i in np.unique(l)],axis=0))
