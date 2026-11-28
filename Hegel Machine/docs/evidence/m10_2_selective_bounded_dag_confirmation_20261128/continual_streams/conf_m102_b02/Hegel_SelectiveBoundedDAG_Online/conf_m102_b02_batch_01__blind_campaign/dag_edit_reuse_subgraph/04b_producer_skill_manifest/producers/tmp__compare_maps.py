import sys,importlib.util,numpy as np,pickle
from itertools import combinations_with_replacement
p='/holdout_task/'; s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; train=s['train']; x=train['x']; n=len(x)
def load(i):
 sp=importlib.util.spec_from_file_location('node',p+'_frozen_nodes/node_0'+str(i)+'.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
kernel=load(2)._kernel

def features(x,d):
 return np.column_stack([np.ones(len(x))]+[x[:,c].prod(axis=1) for k in range(1,d+1) for c in combinations_with_replacement(range(x.shape[1]),k)])
def make(u,v,d=1,robust=False):
 a=features(u,d); keep=np.ones(len(u),bool)
 for _ in range(6):
  c=np.linalg.lstsq(a[keep],v[keep],rcond=None)[0]
  e=np.linalg.norm(a@c-v,axis=1); k=e<max(.001,3*np.median(e))
  if not robust or np.array_equal(keep,k) or k.sum()<a.shape[1]+4:break
  keep=k
 return lambda x:features(x,d)@c, c, e

def score(z):
 gr=kernel(z,z); vals=[]
 for te in [np.arange(52,72),*np.array_split(np.random.default_rng(407).permutation(n),4)]:
  tr=np.setdiff1d(np.arange(n),te); st=train['traj']; modes=np.linalg.svd(st[tr],full_matrices=False)[2][:2]; y=np.c_[train['y'][tr],st[tr]@modes.T]; g=gr[np.ix_(tr,tr)].copy();g.flat[::len(tr)+1]+=1e-6*np.mean(np.diag(g));pred=gr[np.ix_(te,tr)]@np.linalg.solve(g,y);vals.append([np.mean(abs(pred[:,0]-train['y'][te])),np.mean(abs(pred[:,1:]@modes-st[te]))])
 return np.r_[vals[0],np.max(vals[1:],axis=0)]
a,b,c=[s['panel_'+i] for i in 'abc']; fits={}; base={ 'raw':x }
for name,panel,key in [('A',a,'v'),('B',b,'v'),('C1',c,'v1'),('C2',c,'v2')]:
 for d in [1,2]:
  for rev in [False,True]:
   nm=f'{name}{d}{"inv" if rev else ""}'; u,v=panel['u'],panel[key]; fn,co,e=make(v if rev else u,u if rev else v,d,name=='A');fits[nm]=fn;base[nm]=fn(x);print(nm,'calib median',round(np.median(e),5))
 if name=='A':
  _,co,_=make(a['u'],a['v'],1,True);base['deA']=(x-co[0])@np.linalg.inv(co[1:]);
for d in [1,2]:
 for flip in [False,True]:
  k='v'+str(1+flip); l='v'+str(2-flip);f,_,e=make(c[k],c[l],d); nm=k+'to'+l+str(d);fits[nm]=f;print(nm,'calib median',round(np.median(e),5))
results=[]
for on,z in base.items():
 for nm,f in [('id',lambda x:x),*fits.items()]:
  for scale in [.5,1,2]:
   zz=f(z)*scale; results.append((score(zz),on,nm,scale))
print('ranking by worst of official and additional slices');
for q in sorted(results,key=lambda r:max(r[0]))[:30]:print(q[0].round(5),*q[1:])
for q in results:
 if q[1:] in [('deA','B1',1),('deA','B2',1),('raw','id',1),('deA','id',1)]:print('reference',q)
