import pickle,runpy,numpy as np
from scipy.linalg import cho_factor,cho_solve
D='/holdout_task/'
s=pickle.load(open(D+'public_batch.pkl','rb'))[0]['study']; d=runpy.run_path(D+'_frozen_nodes/node_00.py')['run_node']({'__input__':s}); node=runpy.run_path(D+'node_01.py')['_scalar']; y=d['y']; z=d['z']; ids=np.arange(len(y)); seeds=(92,741,1157)
folds=[ids[52:]]+[j for seed in seeds for j in np.array_split(np.random.default_rng(seed).permutation(len(y)),5)]
views=('x','i','a','b','ib','ij','i1','i2','i2b','x2b','a2')
def pred(K,y,n,noise):
 return y.mean()+K[n:,:n]@cho_solve(cho_factor(K[:n,:n]+noise*np.eye(n)),y-y.mean())
rows={}
for j in folds:
 i=np.setdiff1d(ids,j); n=len(i); zz={k:v[np.r_[i,j]] for k,v in z.items()}; v={k:(zz[k]-zz[k][:n].mean(0))/zz[k][:n].std(0) for k in views}; x=v['x']; Q=1+x@x.T/4; g=Q**2
 w=v['i2b']; d=((w[:,None,:]-w[None,:,:])/1.5)**2; base=g+np.exp(-d.sum(2)/2)+np.exp(-d/2).mean(2)**2
 def rec(k,p):rows.setdefault(k,[]).append(np.mean(abs(p-y[j])))
 p0=node(zz,y[i]); rec(('baseline',),p0)
 for k,w in v.items():
  R=w@w.T/4; mods=(R,(1+R)**2,Q*R,Q**2*R,(1+R)**3)
  for a,C in enumerate(mods):
   for wt in (.1,.4,1.):
    p=pred(base+wt*C,y[i],n,.008); rec((k,a,wt),p)
 for wt in (.1,.4,1.):
  C=np.mean([Q*(v[k]@v[k].T/4) for k in ('i1','i2','i2b','x2b','a2')],axis=0); rec(('panel_aggregate',wt),pred(base+wt*C,y[i],n,.008))
print('key; official; shuffled mean; worst; repeated-seed mean/worst')
def stats(e):return np.r_[e[0],np.mean(e[1:]),max(e),*[f(x) for x in np.array(e[1:]).reshape(3,5) for f in (np.mean,np.max)]]
print(('baseline',),stats(rows[('baseline',)]).round(6))
for k,e in sorted(rows.items(),key=lambda kv:max(kv[1][1:]))[:12]:print(k,stats(e).round(6))
