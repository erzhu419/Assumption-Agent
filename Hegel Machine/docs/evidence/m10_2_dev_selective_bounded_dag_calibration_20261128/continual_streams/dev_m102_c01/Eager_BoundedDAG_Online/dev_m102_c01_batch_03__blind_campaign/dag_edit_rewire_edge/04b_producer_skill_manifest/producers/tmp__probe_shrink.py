from scalar_probe import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
import importlib.util
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/node_01.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
def fit(x,z,a,b,y,mode):
 u,v=np.c_[x,a],np.c_[z,b]; scale=np.maximum(u.std(0),1e-8); u,v=(u-u.mean(0))/scale,(v-u.mean(0))/scale
 n,d=u.shape; I=np.eye(n); S=a@a.T; Sq=b@a.T
 def prep(g):
  delta=(g[:,None]-u)**2
  ds=[delta[:,:,:4].sum(-1),delta[:,:,4:].sum(-1)] if mode%2==0 else [*np.moveaxis(delta,-1,0)]; return np.array(ds),(1+g@u.T/d)**2
 D,P=prep(u); Dq,Pq=prep(v); h=len(D)
 def cov(w,D,P,S):
  t=np.exp(w); Q=t[:h,None,None]*D; G=t[h]*np.exp(-Q.sum(0)); R=t[h+1]*P; T=t[h+2]*S; C=np.full_like(G,t[h+3]); return G+R+T+C,[-G*q for q in Q]+[G,R,T,C]
 def loss(w):
  H,J=cov(w,D,P,S); N=np.exp(w[-1])*I; F=cho_factor(H+N); c=cho_solve(F,y); E=cho_solve(F,I)-np.outer(c,c); reg=(w[:h]+3)**2; return .5*y@c+np.log(np.diag(F[0])).sum()+.15*reg.sum(),.5*np.einsum('ij,kij->k',E,np.array(J+[N]))+np.r_[.30*(w[:h]+3),np.zeros(5)]
 w=minimize(loss,np.r_[np.full(h,-3.),0,-4,-1,-1,-8],jac=True,method='L-BFGS-B',bounds=[(-10,2)]*h+[(-16,5)]*4+[(-14,-3)],options={'maxiter':200}).x
 H,_=cov(w,D,P,S); return cov(w,Dq,Pq,Sq)[0]@cho_solve(cho_factor(H+np.exp(w[-1])*I),y)
for va in [np.arange(52,72),np.arange(20),*[np.arange(k,72,3) for k in range(3)]]:
 tr=np.setdiff1d(np.arange(72),va); p,state,y,v,t,x,q=fold(tr,va); a,b=p['scores'],state@p['modes'].T; x,q=m._coordinates(p['K'],p['Kq'],4); y=p['y_centered']+p['y_mean']; baseline=m._predict(x,q,a,b,y); choices=[fit(x,q,a,b,y,k) for k in range(2)]; combo=(2*baseline+choices[0]+choices[1])/4; print('slice',va[:3],'MAE',np.round([np.mean(abs(f-v)) for f in [baseline,*choices,combo]],6),flush=True)
