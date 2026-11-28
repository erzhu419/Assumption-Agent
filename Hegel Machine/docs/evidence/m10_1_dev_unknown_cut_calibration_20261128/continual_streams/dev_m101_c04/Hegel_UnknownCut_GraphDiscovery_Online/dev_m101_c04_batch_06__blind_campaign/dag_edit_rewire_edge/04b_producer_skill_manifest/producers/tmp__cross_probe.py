from dag_probe import *
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
n=52
x=np.c_[u,z]; c=C['u']; cz=b['_affine'](c)@B; cu=(c-B[0])@np.linalg.inv(B[1:]);
for label,K in [('panel',c),('train',u)]:
 V=np.c_[C['v1'],C['v2']] if label=='panel' else np.c_[Y,R[:,-1]]; m=19 if label=='panel' else n
 for name,F in [('linear',basis(K,1)),('quad',basis(K,2)),('odd only',np.c_[basis(K,1),basis(K,3)[:,15:]]),('tanh+quad',np.c_[basis(K,2),np.tanh(K)]),('tanh+adjacent',np.c_[np.ones(len(K)),np.tanh(K),K*np.roll(K,-1,axis=1)]),('tanh+affine',np.c_[basis(K,1),np.tanh(K)]),('tanh+diag',np.c_[np.ones(len(K)),np.tanh(K),K**2]),('cub+tanhs',np.c_[basis(K,3),np.tanh(K)])]:
  print(label,name,fit(F,V,m))
for p in range(2,7):
 F=basis(u,p); avg=np.sqrt(np.mean(u[:n]**2)); degrees=np.concatenate([np.full(len(list(combinations_with_replacement(range(4),k))),k) for k in range(p+1)]); F=F/avg**degrees
 for weight in [0.4,0.7,1.]:
  G=F*weight**np.maximum(degrees-2,0)
  for lam in [.001,.03,.3,3]:
   pred=G@np.linalg.solve(G[:n].T@G[:n]+np.diag(lam*(degrees>=3)+1e-8),G[:n].T@Y[:n]); print('poly',p,weight,lam,np.round([np.mean(abs(pred[:n]-Y[:n])),np.mean(abs(pred[n:]-Y[n:]))],5))
Q=basis(u,2)
print('quad Y and trajectory by horizon',np.linalg.lstsq(Q[:n],np.c_[Y,R][:n],rcond=None)[0].round(3))
print('panel v1 and v2 reduced residuals')
for j in range(8):
 V=np.c_[C['v1'],C['v2']][:,j]
 for pname,pc in [('direct',c),('mixed',cz),('unmixed',cu)]:
  def matrix(k,kind):
   N=np.roll(pc,-1,axis=1)
   return {'tanh':np.tanh(k*pc),'sin':np.sin(k*pc),'rat':pc/(1+k*pc**2),'rat interaction':pc*N/(1+k*pc**2),'tanh interaction':np.tanh(k*pc*N),'rat square':pc**2/(1+k*pc**2)}[kind]
  best=[]
  for kind in ['tanh','sin','rat','rat interaction','tanh interaction','rat square']:
   for base in [basis(c,1),np.c_[basis(c,1),c*np.roll(c,-1,axis=1)],basis(c,2)]:
    def cv(k,stop=19): return fit(np.c_[base,matrix(k,kind)],V,stop)
    k=minimize_scalar(lambda k:cv(k)[0],bounds=(.25,2.5),method='bounded').x
    tr,te=cv(k); best.append((te,tr,kind,base.shape[1],round(k,3)))
  print(j,pname,sorted(best)[:2])
