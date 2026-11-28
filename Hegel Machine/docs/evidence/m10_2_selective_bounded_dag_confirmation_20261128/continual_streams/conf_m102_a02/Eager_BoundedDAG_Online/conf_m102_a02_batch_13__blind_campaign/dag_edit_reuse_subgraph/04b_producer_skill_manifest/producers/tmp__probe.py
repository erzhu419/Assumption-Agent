import numpy as np,pickle,sys,importlib.util
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
print('keys',s.keys()); x=s['train']['x']; y=np.column_stack([s['train']['y'],s['train']['traj']]); print('sv y',np.linalg.svd(y,compute_uv=False).round(4))
def poly(a,d):
 return np.column_stack([np.prod(a[:,p],axis=1) for k in range(d+1) for p in combinations_with_replacement(range(a.shape[1]),k)])
def cv(q):
 e=[]
 for v in [np.arange(52,72),*np.array_split(np.random.default_rng(9).permutation(len(q)),4)]:
  t=np.setdiff1d(np.arange(len(q)),v); r=q[v]@np.linalg.lstsq(q[t],y[t],rcond=1e-8)[0]-y[v];e.append([abs(r[:,0]).mean(),abs(r[:,1:]).mean()])
 return np.round(e,4).tolist()
from scipy.optimize import least_squares
from scipy.special import erf
for name in 'abc':
 p=s['panel_'+name]; u=p['u'];
 for key,v in p.items():
  if key=='u': continue
  print(name,key,'norm percentiles',np.percentile(np.linalg.norm(v,axis=1),[0,25,50,75,90,100]),'row firsts',np.c_[u,v][:6].round(3))
  if name=='a':
   for i in range(4):
    print('axis',i)
    for j in range(4):
     if i==j: continue
     a,b=u[:,i],u[:,j]; q=poly(np.c_[a,b],3); c=least_squares(lambda c:q@c-v[:,i],np.zeros(q.shape[1]),loss='cauchy',f_scale=.02,max_nfev=1000).x
     r=abs(q@c-v[:,i]); print(j,np.percentile(r,[25,50,75,90,95,100]).round(4),c.round(3))
  else:
   aff=poly(u,1); quad=poly(u,2)
   fs={'lin':aff,'poly2':quad}
   args=[u,np.roll(u,1,axis=1)*u,np.roll(u,2,axis=1)*u,u*u,np.roll(u,1,axis=1)+u,np.roll(u,1,axis=1)-u]
   for j,a in enumerate(args):
    for nm,f in [('tanh',np.tanh),('erf',erf),('sin',np.sin),('cos',np.cos)]:
     for w in [0.7,1.25,2,2.15,3,4,5,6.28]:fs[f'{j}:{nm}:{w}']=np.c_[aff,f(w*a)]
   sc=[]
   for nm,q in fs.items():
    t=np.linalg.svd(q,full_matrices=False)[0][:,:np.linalg.matrix_rank(q)]; r=(v-t@(t.T@v))/(1-(t*t).sum(1))[:,None];sc.append((round(np.sqrt((r*r).mean()),5),nm))
   print('best',sorted(sc)[:16]); print('quad coef',np.linalg.lstsq(quad,v,rcond=None)[0].round(3))
