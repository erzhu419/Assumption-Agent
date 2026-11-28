exec(open('/tmp/readout_diag.py').read().split('for keys in')[0])
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import svd
import itertools
print('times',s['eval_times']); U,S,V=svd(tr,full_matrices=False); T=np.c_[y, U[:,:2]*S[:2]]; print('range',T.min(0),T.max(0)); print('z range',z.min(0),z.max(0)); print('y/first mode residual',np.std(y-m._affine(T[:,1:])@m._fit(m._affine(T[:,1:]),y)))
blocks={'z':z,'q':z*z,'p':np.column_stack([z[:,i]*z[:,j] for i,j in itertools.combinations(range(4),2)]),'b':b,'c':c1,'d':c2,'s':np.sin(2.84*z),'h':np.cos(2.6*z)}
def score(pred,iv):
 return [np.mean(abs(pred[:,0]-y[iv])),np.mean(abs(pred[:,1:]@V[:2]-tr[iv]))]
for keys in ['z','zb','zbcd','zbd','zqpsh']:
 X=np.column_stack([blocks[k] for k in keys]); X=(X-X.mean(0))/np.maximum(X.std(0),1e-6); D=m._affine(X); print('basis',keys,'rank',np.linalg.matrix_rank(D,tol=1e-6));
 for kind in ['tanh','tanh+affine','tanh+linearz','tanh_shared','sin+affine']:
  scores=[]
  for it,iv in folds:
   di=D[it]; target=T[it]/T[it].std(0); a=m._fit(di,target); d=di.shape[1]; lin=d if kind.endswith('affine') else (5 if kind.endswith('linearz') else 1); func=np.sin if kind.startswith('sin') else np.tanh
   if kind=='tanh_shared':
    def pred(p,D): return D@p[d:d+3][None,:] if False else p[d:d+3]+np.tanh(D@p[:d])[:,None]*p[d+3:d+6] + D[:,:5]@p[d+6:].reshape(5,3)
    p=np.r_[a[:,1],np.zeros(3),[1,1,0],np.zeros(15)];
   else:
    def pred(p,D): return D[:,:lin]@p[:lin*3].reshape(lin,3)+func(D@p[lin*3:-3].reshape(d,3))*p[-3:]
    p=np.r_[np.zeros(lin*3),a.ravel(),[1.5]*3]
   def fun(p):return np.r_[(pred(p,di)-target).ravel(),.035*p]
   fit=least_squares(fun,p,max_nfev=100); scores.append(score(pred(fit.x,D[iv])*T[it].std(0),iv))
  print(kind,np.round(scores[0],4),np.round(np.max(scores,axis=0),4),flush=True)
 for gam in [.02,.08,.2,.5]:
  rad=cdist(X,X,'sqeuclidean')/X.shape[1]; K=np.exp(-gam*rad)
  for p in [1e-4,.003,.03]:
   vals=[score(K[np.ix_(iv,it)]@np.linalg.solve(K[np.ix_(it,it)]+p*np.eye(len(it)),T[it]),iv) for it,iv in folds]; print('rbf',gam,p,np.round(vals[0],4),np.round(np.max(vals,axis=0),4))
