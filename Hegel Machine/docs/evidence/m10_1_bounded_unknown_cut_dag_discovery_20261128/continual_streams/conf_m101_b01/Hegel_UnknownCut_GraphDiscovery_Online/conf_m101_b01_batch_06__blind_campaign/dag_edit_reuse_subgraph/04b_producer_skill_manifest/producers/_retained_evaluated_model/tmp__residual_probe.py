import pickle
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import svd
np.set_printoptions(precision=4,suppress=True,linewidth=150)
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']
def load(path):
 ns={}; exec(compile(open(P+path).read(),path,'exec'),ns); return ns['run_node']
a=load('_frozen_nodes/node_00.py')({'__input__':s}); b=load('_frozen_nodes/node_01.py')({'node_00':a})
n=len(s['train']['x']); A=a['calibrated_x'][:n]; B=b['x'][:n]
u=s['panel_c']['u']; v=s['panel_c']['v1']; w=s['panel_c']['v2']
i,j=np.triu_indices(4,1)
def quad(x): return np.c_[np.ones(len(x)),x*x,x[:,i]*x[:,j]]
q=np.linalg.lstsq(quad(u),v,rcond=None)[0]; C=quad(B)@q
y=s['train']['y']; t=s['train']['traj']; F=np.c_[y,t]
# Use the four nonzero cyclic interactions identified independently on the panel.
def cyc(x): return np.c_[np.ones(len(x)),x*np.roll(x,-1,axis=1)]
C=cyc(B)@np.linalg.lstsq(cyc(u),v,rcond=None)[0]
def design(a,c): return np.c_[np.ones(len(a)),a,c,(a[:,:,None]*c[:,None,:]).reshape(len(a),-1)]
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
one=np.ones(n); P=np.roll(A,-1,axis=1)*C
base=np.c_[one,C,P.sum(1)]
folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(9).permutation(n),5))+[np.argsort(np.linalg.norm(a,axis=1))[-14:] for a in [A,B,C]]
folds += [np.argsort(np.abs(a[:,j]))[-12:] for a in [A,C] for j in range(4)]
mods={}
for tag,x in [('A',A),('B',B),('C',C),('P',P)]:
 for nm,f in [('square',lambda z:z*z),('cos',np.cos),('cube',lambda z:z**3)]:
  for j in range(4):mods[f'{tag}{j} {nm}']=f(x[:,j:j+1])
  mods[f'{tag} {nm}']=f(x)
 mods[f'{tag} norm2']=(x*x).sum(1)[:,None]
mods.update({f'C{i}C{j}':C[:,i:i+1]*C[:,j:j+1] for i in range(4) for j in range(i+1,4)})
mods['shift0 Csum']=(A*C).sum(1)[:,None];mods['shift2 Csum']=(np.roll(A,-2,axis=1)*C).sum(1)[:,None];mods['shift3 Csum']=(np.roll(A,-3,axis=1)*C).sum(1)[:,None]
mods['none']=np.empty((n,0))
def fit(x,g):return np.linalg.lstsq(x,g,rcond=None)[0]
def calc(X,tr,te):return np.abs(F[te]-X[te]@fit(X[tr],F[tr]))
for tag,j in [('scalar',0),('trajectory',1)]:
 print(tag)
 rows=[]
 for name,vv in mods.items():
  X=np.c_[base,vv]; train=np.mean(np.abs(F-X@fit(X,F)),axis=0)
  es=np.array([calc(X,np.setdiff1d(np.arange(n),te),te).mean(0) for te in folds]); z=lambda xx:xx[...,0] if j==0 else xx[...,1:].mean(-1); e=z(es)
  rows.append((name,z(train),e[0],np.mean(e[1:6]),np.max(e[6:9]),np.max(e[6:]),np.mean(e[6:])))
 for line in [x for x in rows if x[0]=='none']+sorted(rows,key=lambda x:x[5])[:10]+sorted(rows,key=lambda x:x[1])[:5]:print(line[0],np.round(line[1:],5))
 er=[]; names=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); opts=[]
  for name,vv in mods.items():
   X=np.c_[base,vv]; z=F[tr,0:1] if j==0 else F[tr,1:]; d=X[tr]; r=z-d@fit(d,z); lev=(np.linalg.svd(d,full_matrices=False)[0]**2).sum(1)
   opts.append((np.mean(abs(r/(1-lev[:,None]))),name))
  name=min(opts)[1]; names.append(name); e=calc(np.c_[base,mods[name]],tr,te); er.append((e[:,0] if j==0 else e[:,1:]).mean())
 print('fold-internal selection',names,np.round(er,5))
