from panel6 import *
from scipy.optimize import least_squares
from scipy.special import expit
import importlib.util
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m101_b01_batch_05_solver.py'); old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
w=old._calibration(a); ar=np.linalg.norm(np.einsum('nki,ki->nk',old._cal_design(a['u']),w)-a['v'],axis=1); print('calibration',w.round(5),'residual',np.quantile(ar,[0,.5,.9,.94,1]))
dx=old._inverse_calibration(tr['x'],w); print('inverse residual',np.max(abs(np.einsum('nki,ki->nk',old._cal_design(dx),w)-tr['x'])),'ranges',dx.min(0),dx.max(0))
def fitop(p,k,F,bounds):
 fn=lambda x,z:np.c_[af(x),F(z*x)]
 z,e,w=fitfun(p,k,fn,*bounds);print('op',k,F.__name__,z,e);return lambda x:fn(x,z)@w
B=fitop(b,'v',erf,(.6,2.8)); C2=fitop(c,'v2',np.cos,(.6,2.8)); wc=ls(np.c_[np.ones(len(c['u'])),cross(c['u'])],c['v1']); mask=np.linalg.norm(wc[1:],axis=1)>.02; print('C1 terms',mask)
def C1(x):return np.c_[np.ones(len(x)),cross(x)]@wc
ops={'B':B,'C1':C1,'C2':C2}
folds=[(np.setdiff1d(np.arange(52),i),i) for i in np.array_split(np.random.default_rng(624).permutation(52),4)]
def rank(D):
 E=np.concatenate([abs(D[j]@ls(D[i],Y[i])-Y[j]) for i,j in folds]); return np.array([E[:,0].mean(),E[:,1:].mean()])
res=[]
def record(name,D):
 if D.shape[1]<=25 and np.max(abs(D))<1e6: res.append((tuple(rank(D)),name,score(D)))
for start,X in [('raw',tr['x']),('inv',dx),('fwd',np.einsum('nki,ki->nk',old._cal_design(tr['x']),w))]:
 nodes={start:X}
 for depth in range(3):
  for name,x in list(nodes.items()):
   if name.count('/')==depth:
    for op,f in ops.items():
     if op not in name: nodes[name+'/'+op]=f(x)
 for name,x in nodes.items():
  pair=cross(x); sq=x*x
  for nm,D in [('aff',af(x)),('cross',af(pair)),('cycle',af(pair[:,mask])),('diagq',np.c_[af(x),sq]),('quad',np.c_[af(x),sq,pair]),('diagc',np.c_[af(x),sq,x**3]),('tanh',af(np.tanh(x))),('erf',af(erf(x))),('sin',af(np.sin(x))),('cos',af(np.cos(x))),('tanh+aff',np.c_[af(x),np.tanh(x)]),('sin+cos',np.c_[af(np.sin(x)),np.cos(x)])]:record(name+' '+nm,D)
  for op in ops:
   nxt=name+'/'+op
   if nxt in nodes:record(nxt+' skip',np.c_[af(x),nodes[nxt]])
  if name+'/C1' in nodes and name+'/C2' in nodes:
   v1,v2=nodes[name+'/C1'],nodes[name+'/C2']; record(name+' parallel',np.c_[af(v1),v2]);record(name+' parallel+input',np.c_[af(x),v1,v2]);record(name+' parallel+multiply',np.c_[af(v1),v2,v1*v2])
for side in range(2):
 print('rank output',side)
 for e,n,sco in sorted(res,key=lambda r:r[0][side])[:17]: print(n,np.round(e,5),sco)
