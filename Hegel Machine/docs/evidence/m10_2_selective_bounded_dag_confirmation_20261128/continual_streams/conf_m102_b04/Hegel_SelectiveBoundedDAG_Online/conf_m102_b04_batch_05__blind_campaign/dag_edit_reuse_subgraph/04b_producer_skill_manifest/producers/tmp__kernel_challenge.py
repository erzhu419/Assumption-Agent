import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations

def structured(q,n,y,kind='pair',linear=True):
    m=y.mean(); sd=max(y.std(),1e-12); u=(y-m)/sd
    q=(q-q[:n].mean(0))/np.maximum(q[:n].std(0),1e-10)
    d=q.shape[1]; D=(q[:,None]-q[None,:n])**2
    groups=np.asarray([np.ones(d),*np.eye(d),*([np.eye(d)[i]+np.eye(d)[j] for i,j in combinations(range(d),2)] if kind=='pair' else [])]); ids=np.r_[0,np.ones(d,dtype=int),np.full(len(groups)-d-1,2,dtype=int)].astype(int); ng=ids.max()+1
    L=1+q@q[:n].T if linear else np.ones((len(q),n))
    def kernel(D,L,p,grad=False):
        e=-D*np.exp(p[:d]); G=np.exp(e@groups.T+p[d:d+ng][ids]); lin=L*np.exp(p[-2]); K=G.sum(-1)+lin
        return (K,G,e,lin) if grad else K
    def objective(p):
        K,G,e,lin=kernel(D[:n],L[:n],p,True); noise=np.exp(p[-1]); K.flat[::n+1]+=noise; fac=cho_factor(K,check_finite=False); alpha=cho_solve(fac,u,check_finite=False); W=(cho_solve(fac,np.eye(n),check_finite=False)-np.outer(alpha,alpha))*.5
        r=np.einsum('ij,ijk->k',W,G); dr=np.einsum('ij,ijk,ijk->k',W,G@groups,e)
        return .5*u@alpha+np.log(fac[0].diagonal()).sum(),np.r_[dr,np.bincount(ids,weights=r,minlength=ng),np.sum(W*lin),np.trace(W)*noise]
    p=minimize(objective,np.r_[np.zeros(d),np.full(ng,-2.),-2.,-7.],jac=True,method='L-BFGS-B',bounds=[(-12,6)]*d+[(-12,5)]*(ng+1)+[(-13,-2)],options={'maxiter':140}).x
    K=kernel(D,L,p); return m+sd*(K[n:]@cho_solve(cho_factor(K[:n]+np.exp(p[-1])*np.eye(n)),u))

if __name__=='__main__':
 exec(open('/tmp/panel_scan.py').read().split('for name,q in Q.items():')[0])
 import runpy
 base=runpy.run_path('/holdout_task/node_01.py')['_gp']
 ids=np.arange(len(x)); ps={}; slices=[ids[-20:],ids[:20],ids[26:46],*[np.random.default_rng(j).permutation(len(x))[:20] for j in (726,11,82,112)]]; keys=['raw','Av','Bwv','Xw'];
 for i,va in enumerate(slices):
  ti=np.setdiff1d(ids,va); idx=np.r_[ti,va]; n=len(ti); _,_,vt=np.linalg.svd(z[ti],full_matrices=False); b=vt[:2]; yy=z[ti]@b.T
  p={'base':np.column_stack([base(x[idx],n,y) for y in yy.T])@b}
  for key in keys:
   q=Q[key][idx]; p[key+'gp']=np.column_stack([base(q,n,y) for y in yy.T])@b
   for kind in (['joint','pair'] if key=='raw' else ['joint']):
    p[key+kind]=np.c_[structured(q,n,yy[:,0],kind),base(x[idx],n,yy[:,1])]@b
  for names in [('base','Avgp'),('base','Avgp','Bwvgp','Xwgp'),('base','rawpair','Avgp'),('base','rawpair'),('base','rawjoint','Avjoint','Bwvjoint'),('base','rawjoint','rawpair','Avgp','Bwvgp','Xwgp')]: p['+'.join(names)]=np.mean([p[k] for k in names],axis=0)
  for k,v in p.items():ps.setdefault(k,[]).append(np.mean(abs(v-z[va])))
 for k,v in sorted(ps.items(),key=lambda kv:max(kv[1])):print(k,'mean',round(np.mean(v),5),'worst',round(max(v),5),np.round(v,5).tolist())
