import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import *
from scipy.optimize import least_squares, minimize
from scipy.linalg import cho_factor,cho_solve
import pickle
from itertools import combinations_with_replacement as cr,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; g=t['traj']; z,b,e,C,W=features(s,t['x']); n=len(z); p=P(b); c=D(p)@C; eb=E(b)@W
splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),i),i) for i in np.array_split(np.random.default_rng(88).permutation(n),4)]

def mon(x,k):return np.column_stack([np.prod(x[:,j],axis=1) for j in cr(range(x.shape[1]),k)])
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def evaluate(A,reg=1e-5):
 err=[]; Tr=[]
 for i,j in splits:
  H=A/(np.maximum(.04,np.sqrt(np.mean(A[i]**2,axis=0)))); M=H[i]@H[i].T; R=H@H[i].T@np.linalg.solve(M+reg*np.eye(len(i)),y[i]); err.append(abs(R[j]-y[j]).mean()); Tr.append(np.std(R[i]-y[i]));
 return np.round(err,5),np.round(Tr,5)
for x in [z,b,p,np.c_[z,p],np.c_[z,b],np.c_[b,p],np.c_[z,b,p],np.c_[z,c,eb]]:
 for k in (1,2,3):
  A=np.c_[D(x),*[mon(x,v) for v in range(2,k+1)]]; print('pol',x.shape[1],k,evaluate(A),flush=True)

# Estimate physically grouped distances by marginal likelihood to avoid holdout tuning.
def gp(parts,deg):
 for i,j in splits:
  H=[h/max(.025,np.sqrt(np.mean((h[i]-h[i].mean(0))**2))) for h in parts]; H=[h-h[i].mean(0) for h in H]; G=np.array([h@h.T/h.shape[1] for h in H]); ds=np.maximum(G.diagonal(axis1=1,axis2=2)[:,:,None]+G.diagonal(axis1=1,axis2=2)[:,None,:]-2*G,0)
  nd=len(H); u=ds[:,i][:,:,i]; L=G[:,i][:,:,i]; eye=np.eye(len(i)); ym=y[i]-y[i].mean()
  def model(q,U,L,I):
   w=np.exp(q); A=np.einsum('i,ijk->jk',w[:nd],U); R=w[-3]*np.exp(-A); Q=1+np.einsum('i,ijk->jk',w[nd:2*nd],L); K=Q**deg+R+w[-2]*I
   return K,R,Q,w
  def objective(q):
   K,R,Q,w=model(q,u,L,eye); f=cho_factor(K); a=cho_solve(f,ym); J=.5*(cho_solve(f,eye)-np.outer(a,a)); d=np.r_[-w[:nd,None,None]*R*u, w[nd:2*nd,None,None]*L*deg*Q**(deg-1),(R)[None],(w[-2]*eye)[None]]; return .5*(ym@a)+np.log(f[0].diagonal()).sum(),np.einsum('ijk,jk->i',d,J)
  def obj(q):
   v,d=objective(np.r_[q,0.]);return v,d
  q=minimize(obj,np.r_[np.full(2*nd,-2.),-2.,-6.],jac=True,method='L-BFGS-B',bounds=[(-14,4)]*(2*nd+2)).x; K=model(np.r_[q,0.],ds,G,np.eye(n))[0]; pred=y[i].mean()+K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)],ym); print('gp',deg,round(abs(pred-y[j]).mean(),5),np.exp(q).round(4),flush=True)
for parts in ([z,b,p],[z,b,p,eb,e,D(P(b+.5*e))@C],[*z.T[:, :,None],*b.T[:,:,None],*p.T[:,:,None]]):
 gp(parts,2)
