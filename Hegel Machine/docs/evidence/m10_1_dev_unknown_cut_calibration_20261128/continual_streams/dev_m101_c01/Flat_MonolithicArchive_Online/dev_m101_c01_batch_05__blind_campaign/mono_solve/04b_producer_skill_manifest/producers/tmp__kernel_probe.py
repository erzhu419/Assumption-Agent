from probe import *
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
rng=np.random.default_rng(18)
splits=[np.arange(56,72),*np.array_split(rng.permutation(72),4),np.argsort(np.linalg.norm(x,axis=1))[-16:]]
outputs=np.c_[y,z@v[:2].T]
for name,w in [('raw',x),('cal',q),('forward',r),('inverse',(q-B[0])@np.linalg.inv(B[1:])),('both',np.c_[r,(q-B[0])@np.linalg.inv(B[1:])])]:
 d=(w[:,None]-w[None,:])**2; p2=poly(q,2); P=p2@p2.T
 for kind in ('iso','ard','add','mix'):
  errors=[]; pars=[]
  for ids in splits:
   it=np.setdiff1d(np.arange(len(x)),ids); pred=[]
   for col in range(3):
    target=outputs[it,col]; n=w.shape[1] if kind!='iso' else 1
    def k(h):
     d2=d*np.exp(h[:n]); iso=np.exp(-d2.sum(-1)); add=np.exp(-d2).mean(-1)
     return P+np.exp(h[n])*(add if kind=='add' else .5*(iso+add) if kind=='mix' else iso)
    def objective(h):
     K=k(h)[np.ix_(it,it)]; K.flat[::len(it)+1]+=np.exp(h[-1]); f=cho_factor(K); return .5*target@cho_solve(f,target)+np.log(np.diag(f[0])).sum()+.15*np.sum((h[:n]-h[:n].mean())**2)
    h=minimize(objective,[-1.5]*n+[-1,-8],bounds=[(-7,2)]*n+[(-8,4),(-13,-3)],method='L-BFGS-B',options={'maxiter':55}).x
    K=k(h); f=cho_factor(K[np.ix_(it,it)]+np.exp(h[-1])*np.eye(len(it))); pred.append(K[np.ix_(ids,it)]@cho_solve(f,target)); pars.append(h)
   pred=np.array(pred).T; e=abs(pred[:,0]-y[ids]); state=pred[:,1:]@v[:2]; errors.append([e.mean(),abs(state-z[ids]).mean(),abs(state[:,-1]-z[ids,-1]).mean()]);
  print(name,kind,'gate',np.round(errors[0],5),'mean',np.round(np.mean(errors,0),5),'worst',np.round(np.max(errors,0),5),'params',np.round(pars[:3],2).tolist(),flush=True)
