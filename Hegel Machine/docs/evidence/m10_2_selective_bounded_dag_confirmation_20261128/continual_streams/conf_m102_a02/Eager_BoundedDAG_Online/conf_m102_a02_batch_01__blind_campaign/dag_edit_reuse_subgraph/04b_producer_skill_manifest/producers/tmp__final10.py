import importlib.util
from itertools import combinations
import numpy as np,pickle
from scipy.optimize import minimize_scalar
path='/holdout_task/'
def load(f):
 s=importlib.util.spec_from_file_location('m',path+f); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cal=load('node_01.py')._calibrate
s=pickle.load(open(path+'public_batch.pkl','rb'))[0]['study']; b,g=cal(s['panel_a']); x=(s['train']['x']-b)/g; Y=np.c_[s['train']['y'],s['train']['traj']]; N=np.arange(len(x)); pairs=list(combinations(range(x.shape[1]),2));
folds=[N[52:]]+list(np.array_split(np.random.default_rng(17).permutation(N),6))+[j for k in range(4) for j in (np.argsort(x[:,k])[:12],np.argsort(x[:,k])[-12:])]; newer=list(np.array_split(np.random.default_rng(3184).permutation(N),6))+[j for k in range(4) for j in (np.argsort(x[:,k])[:9],np.argsort(x[:,k])[-9:])]
def features(z,p,family):
 if family=='sin':w=np.sin(p*z)**3
 if family=='rational':w=z**3/(1+p*z*z)
 if family=='quartic':w=z**3/(1+p*z**4)
 return np.column_stack((np.ones(len(z)),w,z,w*w,*[w[:,i]*w[:,j] for i,j in pairs]))
def solve(z,U,family,diag,cross):
 scale=np.maximum(U.std(0),1e-8)
 def prof(p):
  A=features(z,p,family); d=A.std(0)**2; n=z.shape[1]; d[1:1+2*n]*=.0135;d[1+2*n:1+3*n]*=diag;d[1+3*n:]*=cross
  W=np.linalg.solve(A.T@A+np.diag(d),A.T@U);return A,W
 def obj(p):
  A,W=prof(p);return np.mean(((A@W-U)/scale)**2)
 p=minimize_scalar(obj,bounds={'sin':(.5,1.2),'rational':(.1,1.7),'quartic':(.02,.8)}[family],method='bounded',options={'xatol':1e-5}).x
 A,W=prof(p)
 if U.shape[1]>1:
  _,_,vt=np.linalg.svd(A@W,full_matrices=False);P=vt[:2].T;W=(W@P)@P.T
 return p,W
if __name__=='__main__':
 for sl in [slice(0,1),slice(1,None)]:
  rs=[]
  for family in ['sin','rational','quartic']:
   for d in [.3,.7,1.4,2.8,5.6]:
    for c in [.2,.7,2.,6.,20.]:
     e=[]
     for idx in folds+newer:
      i=np.setdiff1d(N,idx);p,W=solve(x[i],Y[i,sl],family,d,c);err=abs(features(x[idx],p,family)@W-Y[idx,sl]);e.append([err.mean(),err.mean(0).max()]);
     e=np.array(e);rs.append((e[:15,0].max(),family,d,c,e[0],e[:7].max(0),e[7:15].max(0),e[15:].max(0)))
  for r in sorted(rs,key=lambda r:r[0])[:10]: print(r)
