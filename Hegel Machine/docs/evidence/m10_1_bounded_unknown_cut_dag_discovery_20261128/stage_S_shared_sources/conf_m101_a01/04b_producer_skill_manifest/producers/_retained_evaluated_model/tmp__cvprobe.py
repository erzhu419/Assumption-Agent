from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from probe import *
from itertools import permutations
Z={'x':x,'A':aff(x)@ops['panel_av'],'B':aff(x)@ops['panel_bv']}; Z['AB']=aff(Z['A'])@ops['panel_bv']; Z['BA']=aff(Z['B'])@ops['panel_av']
Y=np.c_[y,j]; rng=np.random.default_rng(37); folds=[(np.arange(52),np.arange(52,72))]; folds += [(np.setdiff1d(np.arange(72),q),q) for q in np.array_split(rng.permutation(72),4)]; q=np.argsort(np.linalg.norm(x,axis=1)); folds += [(q[:-15],q[-15:])]
def cv(F,pen):
 out=[]
 for it,iv in folds:
  sd=F[it].std(0); sd[sd<1e-12]=1; g=F/sd; a=g[it]; coef=np.linalg.lstsq(a,Y[it],rcond=None)[0] if np.max(pen)==0 else np.linalg.solve(a.T@a+np.diag(np.broadcast_to(pen,F.shape[1])),a.T@Y[it]); e=abs(g[iv]@coef-Y[iv]);out.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
 return np.array(out)
ps=[0,.01,.1,1,5,20]; results=[]
for key,z in Z.items():
 for d in (2,3,4,5):
  F=poly(z,d); n=F.shape[1]; w=np.array([0]+[10.**max(0,k-3) for k in range(1,d+1) for _ in combinations_with_replacement(range(4),k)]); w[:5]*=.01
  for p in ps:
   if p==0 and n>52:continue
   a=cv(F,w*p);results.append((key,d,p,a));
 for d in (3,4,5,6):
  F=np.c_[poly(z,2),*[z**k for k in range(3,d+1)]];w=np.r_[0,np.repeat(.01,4),np.ones(10),np.repeat([10.**max(0,k-3) for k in range(3,d+1)],4)];
  for p in ps:
   a=cv(F,w*p);results.append((key,-d,p,a));
for col in range(3):
 print('metric',col)
 for key,d,p,a in sorted(results,key=lambda m:m[3][:,col].max())[:7]: print(key,d,p, 'public + 4 folds + high radius',a[:,col])
print('plain poly2',cv(poly(x,2),0));
U,S,V=np.linalg.svd(j,full_matrices=False);print('temporal vectors',V[:2]);
for key,z in Z.items():
 for d in (1,2,3):
  f=poly(z,d); b=ls(f,Y); print('in-sample',key,d,np.sqrt(np.mean((f@b-Y)**2,axis=0))[[0,1,8]]) if key=='x' else None
