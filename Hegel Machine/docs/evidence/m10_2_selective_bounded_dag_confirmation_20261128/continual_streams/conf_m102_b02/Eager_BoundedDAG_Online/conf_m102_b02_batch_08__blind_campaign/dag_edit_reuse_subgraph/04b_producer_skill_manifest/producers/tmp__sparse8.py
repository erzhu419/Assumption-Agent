from cal8 import *
import pickle,itertools
from scipy.optimize import minimize_scalar
from scipy.linalg import qr
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z,b,c=parts(s); p=s['panel_c']; u,v=p['u'],p['v2']
H=lambda x,k:D(np.c_[x,np.cos(k*x)])
opt=minimize_scalar(lambda k:np.mean((H(u,k)@fit(H(u,k),v)-v)**2),bounds=(.2,4),method='bounded'); k=opt.x; w=fit(H(u,k),v); d=H(z,k)@w
print('C2',k,opt.fun,w.round(5)); tr=s['train']; n=len(tr['y']); ix=np.arange(n); t=s['eval_times']; T=tr['traj']; F=lambda p:-np.expm1(-t[:,None]*p); p=least_squares(lambda p:(F(p)@fit(F(p),T.T)-T.T).ravel(),[1.8,.42],bounds=([1,.15],[3,.85])).x; M=fit(F(p),T.T).T; Y=np.c_[tr['y'],M]; print('modes',p,'corr',np.corrcoef(Y.T).round(4),'RMS',Y.std(0),'trajectory resid',np.abs(F(p)@M.T-T.T).mean());
X={'z':z,'b':b,'c':c,'d':d,'bc':b+c,'bd':b+d,'cd':c+d,'bcd':b+c+d,'b-c':b-c,'b-d':b-d,'c-d':c-d,'.3z+b':.3*z+b,'.3z-b':.3*z-b,'.3z+c':.3*z+c,'.3z+d':.3*z+d,'z+b':z+b,'z+c':z+c,'z+d':z+d,'p':z*np.roll(z,-1,axis=1),'q':np.cos(k*z),'h':np.sign(z)*np.maximum(abs(z)-.35,0)**1.5}
slices=[(np.setdiff1d(ix,it),it) for it in (ix[52:],ix[:20],ix[20:40],ix[40:60])]
def scorepred(a,b):return np.r_[np.abs(a[:,0]-b[:,0]).mean(),np.abs((a[:,1:]-b[:,1:])@F(p).T).mean(axis=0).mean()]
def cv(A,reg=0):
    return np.array([scorepred(A[it]@np.linalg.solve(A[ir].T@A[ir]+reg*np.eye(A.shape[1]),A[ir].T@Y[ir]) if reg else A[it]@fit(A[ir],Y[ir]),Y[it]) for ir,it in slices])
from itertools import combinations_with_replacement
G={}
r=lambda x,j=1:np.roll(x,-j,axis=1)
for a in ['z','b','c','d','h']:
 x=X[a][:n];G[a]=x
 for j in range(3):G[f'{a}sq{j}']=x*r(x,j)
 G[a+'cube']=x**3;G[a+'triple']=x*r(x)*r(x,2)
 for f in [np.sin,np.tanh,np.arcsinh,np.cos]:
  for k in [.7,1.4,2.1,3.5]:G[f'{a}{f.__name__}{k}']=f(k*x)
for a,b1 in itertools.combinations(['z','b','c','d','h'],2):
 for j in range(4):G[f'{a}*{b1}_{j}']=X[a][:n]*r(X[b1][:n],j)
for j in [1,2,3]:
 for k in [.25,.7,1.5,3.]:
  for a in ['z','b','c','d']:
   for v1 in ['z','b','c','d']:
    x=X[a][:n];u=X[v1][:n];G[f'{a}/{v1}{j}/{k}']=x/(1+k*r(u,j)**2)
for a in ['b','c','d']:
 x=X[a][:n]; z1=z[:n]
 for k in [.25,.5,1,2]:
  for sign in [-1,1]:
   u=z1+k*sign*x
   for j in [0,1,2]:G[f'(z+{k*sign}{a})Q{j}']=u*r(u,j)
   G[f'(z+{k*sign}{a})cube']=u**3
base=D(z[:n]); names=list(G); cols=np.stack(list(G.values()),axis=0)
for j,tar in [(0,Y[:,0:1]),(1,Y[:,1:]@F(p)[-1:, :].T)]:
 for ir,it in slices[:2]:
  gs=cols[:,ir]; gs=gs.transpose(0,1,2); yt=tar[ir]; beam=[()]
  for step in range(6):
   hits={}
   for ks in beam:
    A=np.c_[base,*[cols[k] for k in ks]][ir]; Q,sv,_=np.linalg.svd(A,full_matrices=False);Q=Q[:,sv>1e-8]; E=yt-Q@(Q.T@yt); H=gs-Q@(Q.T@gs); U,sv,_=np.linalg.svd(H,full_matrices=False); U*= (sv>1e-7)[:,None,:]; ee=E-U@(U.transpose(0,2,1)@E); lev=1-np.sum(Q*Q,axis=1)-np.sum(U*U,axis=-1); loo=np.mean(abs(ee[:,:,0]/np.maximum(lev,.01)),axis=1)
    for k in np.argsort(loo)[:24]:
     if k not in ks:hits[tuple(sorted((*ks,k)))]=loo[k]
   zr=sorted(hits,key=hits.get)[:18]; beam=zr
   for ks in zr[:2]:
    A=np.c_[base,*[cols[k] for k in ks]];pred=A[it]@fit(A[ir],tar[ir]);print('beam',j,it[0],step,round(hits[ks],4),round(np.mean(abs(pred-tar[it])),4),[names[k] for k in ks],flush=True)
