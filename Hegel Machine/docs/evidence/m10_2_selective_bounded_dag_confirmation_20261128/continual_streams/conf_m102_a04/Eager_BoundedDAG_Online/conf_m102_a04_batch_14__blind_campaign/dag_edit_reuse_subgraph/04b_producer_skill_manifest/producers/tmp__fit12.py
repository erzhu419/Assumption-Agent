import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(f,v):return np.linalg.lstsq(f,v,rcond=1e-10)[0]
def one(u):return np.c_[np.ones(len(u)),u]
def pairs(u):return np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)])
def triples(u):return np.column_stack([np.prod(u[:,j],axis=1) for j in combinations(range(4),3)])
ops={'sin':lambda u,q:np.sin(q*u),'tanh':lambda u,q:np.tanh(q*u),'rat':lambda u,q:u/(1+q*u*u),'soft':lambda u,q:u/(1+q*abs(u)),'atan':lambda u,q:np.arctan(q*u),'erf':lambda u,q:__import__('scipy').special.erf(q*u)}
def scan(u,v,name):
 print(name); out=[]
 for tag,op in ops.items():
  for des in ['none','pair','triple','both','pair_squash','triple_squash','both_squash','pair_rational','cube','pair_cube']:
   if name=='A' and des not in ['none','pair','cube']:continue
   def F(q):
    z=op(u,q); cols=[one(u),z]
    if des.startswith(('pair','both')):cols.append(pairs(z if 'squash' in des else u) if 'rational' not in des else pairs(u)/(1+.22*pairs(u)**2))
    if des.startswith(('triple','both')):cols.append(triples(z if 'squash' in des else u))
    if des.endswith('cube'):cols.append(u**3)
    return np.column_stack(cols)
   def err(q):
    f=F(q);return np.mean((f@ls(f,v)-v)**2)
   r=minimize_scalar(err,bounds=(.07,4),method='bounded'); f=F(r.x); coeff=ls(f,v);res=f@coeff-v; h=np.sum(f*np.linalg.pinv(f).T,axis=1)
   out.append((np.std(res/(1-h[:,None])),np.std(res),tag,des,r.x))
   if name=='A' and des=='none':print(tag,r.x,np.std(res),np.round(coeff.T,3))
 for o in sorted(out)[:17]:print(o)
 for j in range(4):
  if name!='A':break
  print('output',j)
  for tag,op in ops.items():
   ii=[j,(j+2)%4]
   def F(q):return np.c_[one(u),op(u[:,ii],q)]
   def err(q):
    f=F(q);return f@ls(f,v[:,j])-v[:,j]
   fit=least_squares(err,[1.,1.],bounds=(.04,5),gtol=1e-13)
   print(tag,np.std(fit.fun),fit.x,np.round(ls(F(fit.x),v[:,j]),4))
a=s['panel_a'];m=np.linalg.norm(a['v'],axis=1)<5
scan(a['u'][m],a['v'][m],'A')
for k in ['v1','v2']:scan(s['panel_c']['u'],s['panel_c'][k],k)
