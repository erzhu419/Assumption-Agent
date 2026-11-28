from operators import *
from scipy.optimize import minimize_scalar
b,c,d=[ops[k](z) for k in 'BCD'];
def temporal(tau):return -np.expm1(-T/tau)
for i,j in folds:
 L=np.c_[aff(z),b,c,d]; r=Y[i,1:]-L[i]@fit(L[i],Y[i,1:]); est=minimize_scalar(lambda tau: np.linalg.norm(r-(r@temporal(tau))[:,None]*temporal(tau)/np.sum(temporal(tau)**2)),bounds=(.12,30),method='bounded'); print('tau',est.x,'rnorm',est.fun); tt=temporal(est.x); amp=Y[:,1:]@tt/(tt@tt); D=np.c_[L[:72],amp]; w=fit(D[i],Y[i,0]); print('amp contrast',np.mean(abs(D[j]@w-Y[j,0])),w[-1]);
 for nm,F in [('B raw',np.c_[aff(z),b]),('all raw',L),('B+D+D(D)',np.c_[aff(z),b,d,ops['D'](d)]),('B+B²',pol(b,2)),('z+B+B²',np.c_[aff(z),pol(b,2)[:,1:]]),('B²+DD²',np.c_[pol(b,2),pol(ops['D'](d),2)[:,1:]])]:
  r=Y[j]-F[j]@fit(F[i],Y[i]); print(nm,np.mean(abs(r[:,0])),np.mean(abs(r[:,1:])),np.mean(abs(r[:,-1])))
