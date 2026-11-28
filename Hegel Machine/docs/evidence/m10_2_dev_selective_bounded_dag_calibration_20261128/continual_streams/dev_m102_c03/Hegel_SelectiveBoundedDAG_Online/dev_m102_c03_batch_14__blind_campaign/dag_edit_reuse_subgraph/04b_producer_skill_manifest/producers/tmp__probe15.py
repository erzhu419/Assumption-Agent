import numpy as np,pickle
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=150)
def one(*a): return np.column_stack([np.ones(len(a[0])),*a])
def ls(h,y): return np.linalg.lstsq(h,y,rcond=None)[0]
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']; a,b,c,d=u.T; edge=u*np.roll(u,-1,axis=1); pair=np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)]); tri=np.column_stack([np.prod(u[:,j],axis=1) for j in combinations(range(4),3)])
 for k,v in s[p].items():
  if k=='u':continue
  keep=np.linalg.norm(v,axis=1)<4 if p=='panel_a' else np.ones(len(u),bool)
  print(p,k)
  for name,h in [('linear',one(u)),('old',one(u,edge,u**3)),('poly3',one(u,pair,u**2,u**3,edge*u,edge*np.roll(u,-1,axis=1),tri)),('edge',one(edge)),('cos',one(np.cos(2.4*u))),('sin',one(np.sin(2.4*u))),('tanh',one(np.tanh(2.15*u))),('pair',one(pair)),('linedge',one(u,edge)),('fourier',one(np.sin(2.15*u),np.cos(2.15*u))),('polysin',one(u,edge,u**3,np.sin(2.15*u),np.cos(2.15*u)))]:
   h=h[keep]; y=v[keep]; b=ls(h,y); r=y-h@b; lev=np.sum(h*np.linalg.pinv(h).T,axis=1); print(name, np.sqrt(np.mean(r*r)), np.sqrt(np.mean((r/np.maximum(1-lev[:,None],1e-9))**2)))
   if name in ('linear','old') and p=='panel_a': print(b)
