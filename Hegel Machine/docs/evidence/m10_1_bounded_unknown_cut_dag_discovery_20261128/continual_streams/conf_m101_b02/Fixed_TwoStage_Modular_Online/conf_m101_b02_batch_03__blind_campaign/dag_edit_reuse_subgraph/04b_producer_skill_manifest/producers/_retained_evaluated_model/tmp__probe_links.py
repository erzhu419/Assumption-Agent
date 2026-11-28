import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
from sklearn.model_selection import KFold
D=np.load('/tmp/readout_arrays.npz'); Z=D['z']; Y=D['Y']; n=len(Z)

def basis(z,k):
 return np.column_stack([np.ones(len(z)),z]+[np.prod(z[:, j],axis=1) for d in range(2,k+1) for j in combinations_with_replacement(range(4),d)])

def act(h,f):
 if f=='sin': return np.sin(h)
 if f=='tanh': return np.tanh(h)
 if f=='rat': return h/(1+h*h)
 if f=='soft': return h/np.sqrt(1+h*h)
 if f=='atan': return np.arctan(h)
 if f=='exp': return np.exp(-h*h/2)

def train(z,y,f,k,num,seed=2,restarts=10):
 A=basis(z,k); Q=np.linalg.qr(A,mode='reduced')[0]; P=np.eye(len(z))-Q@Q.T; e=P@y; X=np.c_[z,np.ones(len(z))]; rng=np.random.default_rng(seed)
 def err(w):
  H=P@act(X@w.reshape(num,5).T,f); B=np.linalg.lstsq(H,e,rcond=1e-8)[0]; return (H@B-e).ravel()
 best=float('inf')
 for j in range(restarts):
  w=rng.normal(size=(num,5))*[1,1,1,1,.5]; w*= [0.5,1.,2.][j%3]; r=least_squares(err,w.ravel(),bounds=(-12,12),max_nfev=150,ftol=1e-7); v=np.linalg.norm(r.fun)
  if v<best: best=v; W=r.x.reshape(num,5)
 def F(t): return np.c_[basis(t,k),act(np.c_[t,np.ones(len(t))]@W.T,f)]
 C=np.linalg.lstsq(F(z),y,rcond=1e-8)[0]; return F,C,W,best/np.sqrt(e.size)

def score(f,k,num,group):
 es=[]; rs=[]
 for a,b in folds:
  V=np.linalg.svd(Y[a,1:],full_matrices=False)[2][:2]; T=np.c_[Y[:,0],Y[:,1:]@V.T]; std=T[a].std(0); T/=std
  if group=='joint':
   F,C,W,r=train(Z[a],T[a],f,k,num); pred=F(Z[b])@C; rr=[r]
  else:
   ps=[]; rr=[]
   for j in range(3):
    F,C,W,r=train(Z[a],T[a,j:j+1],f,k,num); ps.append(F(Z[b])@C); rr.append(r)
   pred=np.column_stack(ps)
  pred*=std; es.append([np.mean(abs(pred[:,0]-Y[b,0])),np.mean(abs(pred[:,1:]@V-Y[b,1:]))]); rs.append(rr)
 return np.round(es,5).tolist(),np.round(np.mean(rs,0),5).tolist()

if __name__=='__main__':
 V=np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2]; T=np.c_[Y[:,0],Y[:,1:]@V.T]; print('linear and polynomial training RMS',[(k,np.round(np.sqrt(np.mean((T-basis(Z,k)@np.linalg.lstsq(basis(Z,k),T,rcond=None)[0])**2,0)),5)) for k in [1,2,3,4]],flush=True)
 folds=[(np.arange(52),np.arange(52,n)),next(iter(KFold(4,shuffle=True,random_state=87).split(Z)))];
 for k,num,group in [(1,1,'separate'),(2,1,'separate'),(1,2,'joint'),(2,2,'joint'),(1,2,'separate')]:
  for f in ['sin','tanh','rat','atan','exp']:
   print(k,num,group,f,*score(f,k,num,group),flush=True)
