import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
np.set_printoptions(precision=4,suppress=True,linewidth=150)
for panel,keys in [('panel_a',['v']),('panel_c',['v1','v2'])]:
 p=s[panel]; a=p['u']; mask=np.linalg.norm(p[keys[0]],axis=1)<5; a=a[mask]; d=np.c_[np.ones(len(a)),a]; q=a*np.roll(a,-1,axis=1); qq=np.array([a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)]).T
 for k in keys:
  v=p[k][mask]; print(panel,k)
  for tag,b in [('cyc',np.c_[d,q]),('full2',np.c_[d,qq]),('cyc+3',np.c_[d,q,a**3]),('cyc+diag',np.c_[d,q,a**2,a**3]),('diag3',np.c_[d,a**2,a**3])]:
   c=ls(b,v); r=b@c-v; h=(b*np.linalg.pinv(b).T).sum(1); print(tag,np.sqrt(np.mean(r*r,0)),np.sqrt(np.mean((r/(1-h[:,None]))**2,0)))
   if tag in ('cyc','diag3'): print(c)
  if panel=='panel_a':
   for i in range(4):
    y=v[:,i]; print('col',i,'correl resid quad',np.corrcoef(np.c_[q,q**2,q**3,a**3,(np.roll(a,-1,axis=1)*a**2),(a*np.roll(a,-1,axis=1)**2),y-np.c_[d,q]@ls(np.c_[d,q],y)].T)[-1,:-1].reshape(-1,4))
  else:
   for name,fn in [('atanh',lambda x:np.arctanh(x)),('sinh',np.sinh),('cube',lambda x:x**3),('sqrt',lambda x:np.sign(x)*np.sqrt(abs(x))),('asinh',np.arcsinh),('cbrt',np.cbrt)]:
    sc=np.linspace(.05,3,240) if name!='atanh' else np.linspace(.05,.999/abs(v).max(),240)
    for j in range(4):
     r=[np.linalg.norm((lambda y:d@ls(d,y)-y)(fn(v[:,j]*m)))/np.linalg.norm(fn(v[:,j]*m)) for m in sc]; ix=np.argmin(r); print(name,j,round(sc[ix],3),round(r[ix],5),end='; ')
    print()
