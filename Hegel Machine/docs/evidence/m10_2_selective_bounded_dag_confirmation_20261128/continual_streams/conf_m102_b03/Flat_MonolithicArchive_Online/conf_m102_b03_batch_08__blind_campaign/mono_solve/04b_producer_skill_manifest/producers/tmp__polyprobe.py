exec(open('/tmp/graph.py').read().split("print('B'")[0])
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import factorial
from scipy.optimize import least_squares
Y=np.c_[s['train']['y'],s['train']['traj']]; n=52
z2=(x*coef[:,1])+coef[:,0]; B0=ops['B'];P=ops['P'];Q=ops['Q']; I=lambda x:(x-b[0])@np.linalg.inv(b[1:]);
def fit(K,r):
 W=solve(K[:n,:n]+r*np.eye(n),np.c_[Y[:n],np.eye(n)],assume_a='pos'); E=abs(K[n:,:n]@W[:,:9]-Y[n:]); L=abs(W[:,:9]/np.diag(W[:,9:])[:,None]);return np.array([np.mean(E[:,0]),np.mean(E[:,1:]),np.mean(L[:,0]),np.mean(L[:,1:])]);
for key,Z in [('z',z),('x',x),('a',z2),('b',B0(z)),('i',I(z))]:
 print(key)
 for name,W in [('raw',Z),('tanh',np.tanh(1.25*Z)),('sin',np.sin(1.25*Z)),('rat',Z/(1+.22*Z*Z)),('adjcos',np.c_[adj(Z),cs(Z)]),('both',np.c_[Z,adj(Z),cs(Z)])]:
  W=W-W.mean(0); W=W/np.sqrt(np.mean(np.sum(W[:n]**2,axis=1))); G=W@W.T; Dist=cdist(W,W,'sqeuclidean'); R=[]
  for kd,K in [(f'p{d}',(1+g*G)**d) for d in [2,3,4,5,7,10] for g in [.1,1,4]]+[(f'rbf{g}',1+np.exp(-g*Dist)) for g in [.07,.15,.4,.8,1.6]]:
   for r in [1e-7,1e-5,.001,.03]:
    e=fit(K,r);R.append((kd,r,np.round(e,4)))
  print(name,'outer',min(R,key=lambda v:sum(v[2][:2])),'inner',min(R,key=lambda v:sum(v[2][2:])))
# Direct transform plus all raw, pairwise products, orthogonal and mixed Fourier harmonics.
for Zname,Z in [('Z',z),('X',x),('B',B0(z)),('I',I(z))]:
 for w in [1.25,2.4,2.5,np.pi,3.6,4.8,5.0]:
  cols=[]
  for j,k in it.combinations(range(4),2):cols.extend([Z[:,j]+Z[:,k],Z[:,j]-Z[:,k],Z[:,j]*Z[:,k]])
  L=np.c_[Z,np.array(cols).T];F=np.c_[Z,adj(Z),np.sin(w*L),np.cos(w*L)];F-=F.mean(0);F/=F.std(0)+1e-10;K=1+F@F.T/F.shape[1]; print('freq',Zname,w,min((fit(K,r) for r in [.0001,.002,.02,.1]),key=lambda e:sum(e[2:])))
