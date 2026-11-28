from maps15 import *
from itertools import combinations
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True,linewidth=180)
def pair(u):return np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)])
for p,key in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u']; v=s[p][key]; m=np.linalg.norm(v,axis=1)<4; u=u[m];v=v[m]; print(p,key,len(u),u.min(),u.max())
 if p=='panel_a':
  for k in [1.25,2.15,2.4]:
   for base in [one(u,np.sin(k*u)),one(u,np.sin(k*u),np.cos(k*u))]:
    b=ls(base,v); print(k,np.std(v-base@b),b)
 else:
  ds={'u':u,'pair':pair(u),'cubic':u**3, 'edge':u*np.roll(u,-1,axis=1),'sum':u+np.roll(u,-1,axis=1),'diff':u-np.roll(u,-1,axis=1)}
  scores=[]
  for bn,base in [('none',np.empty((len(u),0))),('u',u),('pairs',pair(u)),('both',np.column_stack([u,pair(u)]))]:
   for dn,d in ds.items():
    for fn,f in func.items():
     def err(k):
      h=one(base,f(d,k)); r=v-h@ls(h,v); lev=np.sum(h*np.linalg.pinv(h).T,axis=1);return np.mean((r/(1-lev[:,None]))**2)
     ks=np.linspace(.2,5,35); i=np.argmin([err(k) for k in ks]); k=minimize_scalar(err,bounds=(max(.05,ks[i]-.16),ks[i]+.16),method='bounded').x
     scores.append((np.sqrt(err(k)),bn,dn,fn,k))
  print(*sorted(scores)[:22],sep='\n')
