from dag_probe import *
from itertools import combinations
from scipy.optimize import least_squares
c=C['u']; v=C['v1']; Q=basis(c,2)
print('panel C v2 quadratic coefficients (constant, linear, upper-triangular):\n',np.round(np.linalg.lstsq(Q,C['v2'],rcond=None)[0],3))
for j in range(4):
 D=v[:,j]; print('v1',j)
 for name,g in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan)]:
  for dname,x in [('coordinate',c[:,j:j+1]),('projection',c)]:
   F=basis(c,1); k=x.shape[1]; init=np.r_[np.linalg.lstsq(F,D,rcond=None)[0],1.,0.,np.eye(k)[j if k>1 else 0]]
   def res(p,ids): return (F@p[:5]+p[5]*g(p[6]+x@p[7:])-D)[ids]
   p=least_squares(res,init,args=(np.arange(19),),max_nfev=1000).x
   print(name,dname,round(np.mean(abs(res(p,slice(19,None)))),4),round(np.mean(abs(res(p,slice(19)))),4),np.round(p,2))
xy=np.c_[u,z]; U=basis(u,2); N=52
cube=np.c_[basis(u,3)[:,15:],basis(z,3)[:,15:]]
for nm,F in [('quad',U),('cubic',basis(u,3)),('quad+tanh',np.c_[U,np.tanh(xy)]),('quad+sin',np.c_[U,np.sin(xy)]),('quad+cube diag',np.c_[U,xy**3]),('quad+tanh grid',np.c_[U,*[np.tanh(k*xy) for k in [.5,1,1.5,2]]]),('quad+v1 approx',np.c_[U,np.tanh(z),np.tanh(u),u*np.tanh(z),z*np.tanh(u)])]:
 print('Y',nm,fit(F,Y,N))
for target in [Y,R[:,-1]]:
 F=basis(u,3); co=np.linalg.lstsq(F[:N],target[:N],rcond=None)[0]; print('poly target coefficients',np.round(co,3)); print('fit residual mae',fit(F,target,N))
# Rank candidate single cubic or saturating additions relative to the quadratic residual.
dic=[(f'cube{i}',f) for i,f in enumerate(cube.T)]
for k in [.5,1,1.5,2,3]:
 for i,x in enumerate(xy.T):
  for nm,f in [('tanh',np.tanh(k*x)),('sin',np.sin(k*x)),('rat',x/(1+(k*x)**2))]: dic.append((f'{nm}{i}@{k}',f))
D=np.column_stack([x[1] for x in dic]); target=Y
for step in range(8):
 e=target[:N]-U[:N]@np.linalg.lstsq(U[:N],target[:N],rcond=None)[0]
 Dr=D[:N]-U[:N]@np.linalg.lstsq(U[:N],D[:N],rcond=None)[0]
 i=np.argmax(abs(e@Dr)/np.maximum(1e-12,np.linalg.norm(Dr,axis=0))); U=np.c_[U,D[:,i]]; print('pursuit',dic[i][0],fit(U,target,N))
