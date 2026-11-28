import numpy as np,pickle,runpy
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train'];x=t['x'];y=t['y'];z=t['traj'];vh=np.linalg.svd(z,full_matrices=False)[2][:2];Y=np.c_[y,z@vh.T];Y[:,1:]/=np.sqrt(8); np.set_printoptions(precision=4,suppress=True,linewidth=140)
a=s['panel_a']; aff=lambda q:np.c_[np.ones(len(q)),q]; i=np.argsort(np.linalg.norm(a['u']-a['v'],axis=1))[:-3];M=np.linalg.lstsq(aff(a['u'])[i],a['v'][i],rcond=None)[0];xc=np.linalg.solve(M[1:].T,(x-M[0]).T).T
c=s['panel_c'];U,S,V=np.linalg.svd(np.linalg.lstsq(aff(c['u']),c['v1'],rcond=None)[0][1:]);W=U@V; C=[np.linalg.lstsq(aff(c['u']),c[v],rcond=None)[0] for v in ['v1','v2']];print('correlation',np.corrcoef(np.c_[x,Y].T)); print('y vs amplitudes, sorted by first amplitude:\n',np.c_[Y,x][np.argsort(Y[:,1])]); print('panel c maps',C); print('orthogonal',W)
folds=[(i,j) for r in range(3) for i,j in KFold(6,shuffle=True,random_state=r+12).split(x)]
def err(P,T):
 E=np.abs(P-T);return np.r_[E[:,0].mean(),np.abs((P[:,1:]-T[:,1:])@vh).mean()*np.sqrt(8)]
def ks(K,p):
 E=np.array([err(K[np.ix_(j,i)]@cho_solve(cho_factor(K[np.ix_(i,i)]+p*np.eye(len(i))),Y[i]),Y[j]) for i,j in folds]); return np.r_[E.mean(0),E.max(0)]
R=[]
for nm,q in [('raw',x),('cal',xc),('ortho',xc@W),('C1',aff(xc)@C[0]),('C2',aff(xc)@C[1])]:
 for f in (.5,1,1.5,2,2.5,3,4,5,6,8):
  dist=cdist(q,q)*f; K0=aff(q)@aff(q).T
  S=np.sin(q*f); CC=np.cos(q*f); F=np.c_[S,CC]; G=F@F.T
  for k,K in [('rbf',np.exp(-.5*dist**2)),('m32',(1+dist)*np.exp(-dist)),('m52',(1+dist+dist**2/3)*np.exp(-dist)),('add-rbf',np.exp(-.5*((q[:,None]-q[None,:])*f)**2).sum(2)),('Fourier',G+K0),('Fourier-pair',(G+1)**2+K0),('Fourier-poly',G+K0**2)]:
   for p in (.0001,.003,.03,.2):R.append((ks(K+1,p),nm,k,f,p))
for j in (0,1):
 print('kernel top',j)
 for r in sorted(R,key=lambda h:h[0][j])[:10]:print(r)
for nm,q in [('raw',x),('cal',xc),('ortho',xc@W),('C1',aff(xc)@C[0]),('C2',aff(xc)@C[1])]:
 for d in range(4):
  r=np.argsort(q[:,d]); print(nm,d,'coordinate / responses in order (groups of 8)',np.array([np.c_[q[g,d],Y[g]].mean(0) for g in np.array_split(r,9)]).T)
