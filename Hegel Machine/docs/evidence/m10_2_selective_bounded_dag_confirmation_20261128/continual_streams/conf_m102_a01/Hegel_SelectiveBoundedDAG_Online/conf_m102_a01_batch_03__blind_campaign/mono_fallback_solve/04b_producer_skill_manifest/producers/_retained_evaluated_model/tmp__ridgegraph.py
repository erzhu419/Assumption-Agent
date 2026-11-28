from explore import *
from sklearn.model_selection import KFold
from scipy.optimize import least_squares
n=52
F=poly(z,2)
print('linear trajectory singular',np.linalg.svd(fit(aff(z),R[:,1:])[1:])[1]); print('y regressed on traj+quad');
for j in (1,2):
 for g in [np.c_[z,R[:,1:]],R[:,1:],np.c_[z,T[:,-1]]]:
  Q=poly(g,j); q=fit(Q[:n],y[:n]); print(j,g.shape,np.mean(abs(Q[n:]@q-y[n:])),np.std(Q[:n]@q-y[:n]))
for j in range(3):
 q=fit(F[:n],R[:n,j]); M=np.zeros((4,4)); ix=np.triu_indices(4); M[ix]=q[5:]; M=(M+M.T)/2; print('quadratic modes',j,np.linalg.eigh(M)); print('quad residuals', j, np.sort(abs(F[:n]@q-R[:n,j]))[-10:]);
funcs={'s':np.sin,'t':np.tanh,'r':lambda a:1/(1+a*a),'g':lambda a:np.exp(-a*a),'u':lambda a:a/(1+a*a),'c':np.cos,'q':lambda a:a*a/(1+a*a)}
rng=np.random.default_rng(22)
for j in range(3):
 yy=R[:,j]; base=F if j!=1 else aff(z); modes=np.linalg.svd(fit(poly(z,3)[:n],yy[:n])[15:].reshape(1,-1),full_matrices=False)[0]; out=[]
 for key,fn in funcs.items():
  for k in [1,2]:
   def mat(a):return np.c_[base,fn(z@a.reshape(k,5)[:,:4].T+a.reshape(k,5)[:,4])]
   def res(a):
    M=mat(a)[:n]; return M@fit(M,yy[:n])-yy[:n]
   cand=[]
   for rep in range(12):
    p=least_squares(res,rng.normal(size=k*5)*.9,bounds=(-5,5),max_nfev=110,ftol=1e-7); A=mat(p.x); coef=fit(A[:n],yy[:n]); cand.append((np.std(p.fun),np.mean(abs(A[n:]@coef-yy[n:])),p.x,coef[-k:]))
   print('ridge',j,key,k,min(cand,key=lambda x:x[0]),flush=True)
