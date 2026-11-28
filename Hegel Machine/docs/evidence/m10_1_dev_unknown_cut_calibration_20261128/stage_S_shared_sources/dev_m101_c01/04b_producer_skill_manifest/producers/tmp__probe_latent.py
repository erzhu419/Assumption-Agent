import pickle,numpy as np
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];x=t['x'];Y=np.c_[t['y'],t['traj']];T=s['eval_times'];np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(X,y):return np.linalg.lstsq(X,y,rcond=None)[0]
def pol(x,d):return np.c_[aff(x),*[np.prod(x[:,i],1) for k in range(2,d+1) for i in cr(range(4),k)]]
E=np.c_[-np.expm1(-1.52*T),-np.expm1(-.22*T)]; C=Y[:,1:]@np.linalg.pinv(E).T; Z=np.c_[Y[:,0],C];
for j in range(3):
 w=fit(pol(x,3),Z[:,j]);H=np.zeros((4,4));
 for (i,k),v in zip(cr(range(4),2),w[5:15]):H[i,k]=H[k,i]=v*(1 if i==k else .5)
 print('target',j,'aff',w[:5],'Hessian',(np.linalg.eigh(H)))
 print('cubic singular values',np.linalg.svd(np.array([[[w[15+list(cr(range(4),3)).index(tuple(sorted((i,k,l))))]/(1 if i==k==l else 3 if len({i,k,l})==2 else 6) for l in range(4)]for k in range(4)]for i in range(4)]).reshape(4,16),compute_uv=False))
print('C, y corr',np.corrcoef(Z.T));print('telemetry',s['telemetry_vec'])
# Fit low-rank quadratic with ridge nonlinearities. A small number of ridge directions
# is a falsifiable alternative to fitting all cubic coefficients independently.
F=pol(x,2); W=fit(F,Z);P=np.linalg.pinv(F); R=Z-F@W
fun={'cubic':lambda a:a**3,'sin':lambda a:np.sin(a)-a,'tanh':lambda a:np.tanh(a)-a,'asinh':lambda a:np.arcsinh(a)-a,'rat':lambda a:a/(1+a*a)-a,'cos':lambda a:np.cos(a)-1+.5*a*a}
rng=np.random.default_rng(31)
for name,f in fun.items():
 for n in (1,2,3):
  for j in (0,1,2):
   def design(q,z):return np.c_[pol(z,2),f(z@q.reshape(n,5)[:,:4].T+q.reshape(n,5)[:,4])]
   scores=[];params=[]
   for I,J in [(np.arange(56),np.arange(56,72)),(np.arange(16,72),np.arange(16))]:
    def res(q):
     a=design(q,x[I]);return a@fit(a,Z[I,j])-Z[I,j]
    rs=[least_squares(res,np.c_[rng.normal(size=(n,4))*.6,np.zeros(n)].ravel(),max_nfev=130) for _ in range(3)]
    r=min(rs,key=lambda r:r.cost); a=design(r.x,x);scores.append(np.mean(abs(a[J]@fit(a[I],Z[I,j])-Z[J,j])));params.append(r.x.reshape(n,5))
   if max(scores)<(.019 if j==0 else .050):print(name,n,j,np.array(scores),params[0])
# Scores of the reference quadratic/cubic on the same latent amplitudes.
for d in (2,3):
 for I,J in [(np.arange(56),np.arange(56,72)),(np.arange(16,72),np.arange(16))]:
  a=pol(x,d);print('baseline latent',d,np.mean(abs(a[J]@fit(a[I],Z[I])-Z[J]),axis=0))
