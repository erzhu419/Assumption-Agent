exec(open('/tmp/probe.py').read().split("for p in ['panel_a'")[0])
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement

def pred(w,r,tr,va,kind):
 X=np.c_[np.ones(len(w)),w]; Q,R,VT=np.linalg.svd(X[tr],full_matrices=True); U=Q[:,5:]; L=X[va]@np.linalg.pinv(X[tr]); yy=U.T@r[tr]; w=w-w[tr].mean(0); ds=(w[:,None,:]-w[None,:,:])**2; ps=w[:,None,:]*w[None,:,:]; I=np.eye(len(yy)); sq=np.stack([w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(4),2)],1); g=sq@sq.T
 def kernel(p):
  a=np.exp(p[:4]); D=ds@a; ww=np.exp(p[4:]); k0=np.exp(-D/2); J=-.5*k0; k2=g
  if kind=='mat':
   h=np.sqrt(3*D); k0=(1+h)*np.exp(-h); J=-1.5*np.exp(-h)
  if kind=='add':
   k0=np.exp(-ds*a/2); J=-.5*k0; return ww[0]*k0.sum(2)+ww[1]*np.exp(-D/2)+ww[2]*k2, np.concatenate((ww[0]*J*ds*a-.5*ww[1]*np.exp(-D/2)[:,:,None]*ds*a, (ww[0]*k0.sum(2))[:,:,None],(ww[1]*np.exp(-D/2))[:,:,None],(ww[2]*k2)[:,:,None]),2)
  return ww[0]*k0+ww[1]*k2,np.concatenate((ww[0]*J[:,:,None]*ds*a,(ww[0]*k0)[:,:,None],(ww[1]*k2)[:,:,None]),2)
 def loss(p):
  G,D=kernel(p); K=U.T@G[tr[:,None],tr]@U+np.exp(p[-1])*I; c=cho_factor(K); al=cho_solve(c,yy); E=cho_solve(c,I)-np.outer(al,al); d=.5*np.einsum('ij,ijk->k',U@E@U.T,D[tr[:,None],tr]); pen=.75*np.sum((p[:4]-p[:4].mean())**2); d[:4]+=1.5*(p[:4]-p[:4].mean()); return .5*yy@al+np.log(c[0].diagonal()).sum()+pen, np.r_[d,.5*np.trace(E)*np.exp(p[-1])]
 seed=[-1]*4+[-3]*({'add':3}.get(kind,2))+[-10]; fit=minimize(loss,seed,method='L-BFGS-B',jac=True,bounds=[(-5,3)]*4+[(-16,2)]*(len(seed)-5)+[(-16,-4)]); p=fit.x; G=kernel(p)[0]; K=G[tr[:,None],tr]; al=cho_solve(cho_factor(U.T@K@U+np.exp(p[-1])*I),yy); return L@r[tr]+(G[va[:,None],tr]-L@K)@U@al
if __name__=='__main__':
 rng=np.random.default_rng(801); splits=[np.arange(len(x))]+[rng.permutation(len(x)) for _ in range(8)]; allp=[]
 for p in splits:
  tr,va=p[:n],p[n:]; S={**s,'train':{**s['train'],**{k:s['train'][k][tr] for k in ['x','y','traj']}},'test':{'x':x[va]}}; old=m.solve_batch(S); V=np.linalg.svd(t[tr],full_matrices=False)[2][0]; a=t@V; pp=[]; names=[]
  for kind in ('rbf','mat','add'):
   for k,w in [('x',x),('z',z),('h',h)]:
    pp.append((pred(w,y,tr,va,kind),pred(w,a,tr,va,kind)[:,None]*V)); names.append(kind+k)
  pp.append((old['y_hat'],old['state_pred'])); names.append('old'); pp.append(tuple(np.mean([q[j] for q in pp],0) for j in range(2))); names.append('avg'); print('split',[[name,round(np.mean(abs(q[0]-y[va])),5),round(np.mean(abs(q[1]-t[va])),5)] for name,q in zip(names,pp)],flush=True); allp.append((va,pp))
 import pickle;pickle.dump((names,allp),open('/tmp/ardfit.pkl','wb'))
