from probe import *
from scipy.optimize import least_squares
from scipy.special import expit
names=lambda d:[j for k in range(d+1) for j in combinations_with_replacement(range(4),k)]
def trim(u,v,d):
 a=pol(u,d); rng=np.random.default_rng(424); best=(np.inf,None)
 for _ in range(500):
  ids=rng.choice(len(a),a.shape[1],replace=False); c=fit(a[ids],v[ids]);
  for j in range(4):
   e=np.sum((a@c-v)**2,axis=1); ids=np.argsort(e)[:40]; c=fit(a[ids],v[ids]);
  score=np.mean(np.sort(np.sum((a@c-v)**2,axis=1))[:40]);
  if score<best[0]:best=score,c
 return best
if __name__=='__main__':
 p=s['panel_a']; err,c=trim(p['u'],p['v'],2); print('A trimmed',err,list(zip(names(2),c.tolist()))); print('A residual',np.sort(np.linalg.norm(pol(p['u'],2)@c-p['v'],axis=1)))
 t=s['train']['traj']; U,d,V=np.linalg.svd(t,full_matrices=False); print('time',s['eval_times'],'spectrum',d); print('time basis',V[:4]); print('latent std',np.std(U*d,axis=0))
 for panel,key in [('b','v'),('c','v1')]:
  p=s['panel_'+panel]; u=p['u']; v=p[key]; A=pol(u,1); Q=pol(u,2); cub=pol(u,3); print(panel,key)
  for fn,iv in [('tanh',np.arctanh),('sin',np.arcsin),('asinh',np.sinh),('softsign',lambda x:x/(1-abs(x))),('sig',lambda x:np.log((1+x)/(1-x)))]:
   for deg,design in [(1,A),(2,Q)]:
    base=np.max(abs(v),axis=0); out=[]
    for scale in [1.01,1.25,1.5,2.,3.]:
     z=iv(v/np.maximum(scale,base+.01)); est=design@fit(design,z); out.append(np.mean(abs(est-z)))
    print(fn,deg,np.round(out,4))
  print('odd cubic',np.std(v-np.c_[A,cub[:,15:]]@fit(np.c_[A,cub[:,15:]],v),axis=0)); print('sin coordinate',np.std(v-np.c_[A,np.sin(2*u),np.cos(2*u)]@fit(np.c_[A,np.sin(2*u),np.cos(2*u)],v),axis=0))
