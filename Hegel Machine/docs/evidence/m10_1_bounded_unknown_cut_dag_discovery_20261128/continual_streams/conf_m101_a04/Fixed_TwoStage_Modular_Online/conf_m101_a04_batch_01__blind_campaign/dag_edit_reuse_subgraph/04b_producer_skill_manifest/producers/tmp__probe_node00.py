import pickle, numpy as np
from itertools import combinations, product, combinations_with_replacement
from collections import defaultdict
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a,b,c=(s['panel_'+k] for k in 'abc')
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def mix(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)])
def v2(x):return np.c_[aff(x),np.logaddexp(1.1*x,-1.1*x)-np.log(2)]
f=aff(a['u']); w=np.ones(len(f))
for _ in range(20):
 coef=fit(f*w[:,None],a['v']*w[:,None]); r=np.linalg.norm(a['v']-f@coef,axis=1); w=1/np.sqrt(1+(r/max(1e-10,np.median(r)))**2)
x=s['train']['x']; z=np.linalg.solve(coef[1:].T,(x-coef[0]).T).T
p=aff(z)@fit(aff(b['u']),b['v']); pi=np.linalg.solve(fit(aff(b['u']),b['v'])[1:].T,(z-fit(aff(b['u']),b['v'])[0]).T).T
def c1(z):return np.c_[np.ones(len(z)),mix(z)]@fit(np.c_[np.ones(len(c['u'])),mix(c['u'])],c['v1'])
def c2(z):return v2(z)@fit(v2(c['u']),c['v2'])
y=s['train']['y']; t=s['train']['traj']; n=len(y); rg=np.random.default_rng(566); folds=[(np.arange(n-20),np.arange(n-20,n))]+[(np.setdiff1d(np.arange(n),va),va) for va in np.array_split(rg.permutation(n),4)]
def evaluate(h):
 f=np.c_[np.ones(n),h,.4*np.column_stack([h[:,i]*h[:,j] for i,j in combinations_with_replacement(range(h.shape[1]),2)])]; K=f@f.T
 errors=[]
 for a,b in folds:
  ka=K[np.ix_(a,a)]; kb=K[np.ix_(b,a)]; e=np.eye(len(a)); dy=kb@np.linalg.solve(ka+.0003*e,y[a])-y[b]; dt=kb@np.linalg.solve(ka+.001*e,t[a])-t[b]; errors.append((abs(dy).mean(),abs(dt).mean(),abs(dt).mean(0).max()))
 return np.array(errors)
results=[]
for name,l in [('raw',x),('unscaled',z),('B',p),('Binv',pi)]:
 blocks={'L':l,'U':z,'B':p,'C1':c1(l),'C2':c2(l)}
 for k in range(1,5):
  for keys in combinations(blocks,k):
   for scales in [(1,)*k]+([(1,)+(.5,)*(k-1),(1,)+(2,)*(k-1)] if k>1 else []):
    e=evaluate(np.column_stack([blocks[key]*scale for key,scale in zip(keys,scales)])); results.append((max(e[:,0].max(),e[:,1].max()),name,keys,scales,e))
for _,name,keys,sc,e in sorted(results,key=lambda x:x[0])[:28]:print(name,keys,sc, np.round(e[0],5),np.round(e.max(axis=0),5))
print('best official splits:')
for _,name,keys,sc,e in sorted(results,key=lambda x:max(x[-1][0,:2]))[:10]: print(name,keys,sc,np.round(e[0],5),np.round(e.max(axis=0),5))
