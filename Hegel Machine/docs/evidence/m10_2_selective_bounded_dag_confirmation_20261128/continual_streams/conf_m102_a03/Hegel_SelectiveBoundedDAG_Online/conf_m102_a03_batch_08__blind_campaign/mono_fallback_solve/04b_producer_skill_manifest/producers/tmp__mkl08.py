import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
D=np.load('/tmp/feat08.npz'); z,y,T=[D[k] for k in ['z','y','t']]; E=np.load('/tmp/enc08.npz'); W=np.load('/tmp/bfit08.npz'); N=len(z); rng=np.random.default_rng(555); splits=[np.arange(N)]+[rng.permutation(N) for _ in range(6)]
def ks(X,I):
 X=(X-X[I].mean(0))/X[I].std(0); G=X@X.T/4; D=(X[:,None]-X[None,:])**2
 return G,D,X

def basis(I,variant):
 A,DA,a=ks(z,I); C=[np.ones_like(A),A,A*A]; blocks=[]
 for X in [z-.4*np.sin(f*z) for f in [2.4,2.9,3.3]]+[E['panel_bvsin'],W['f2']]:
  G,R,x=ks(X,I); K=[G,G*G,G*A];
  for L in [1.5,6,24]:
   S=np.exp(-(R+.25*DA)/(1.25*L)); M=S.sum(2); K.extend([M/4,(M*M-(S*S).sum(2))/12]);
   if variant: K.extend([S[:,:,i] for i in range(4)]); K.extend([S[:,:,i]*S[:,:,j] for i in range(4) for j in range(i)])
  blocks.append(K)
 return [np.array(C+K) for K in blocks]

def infer(B,I,J,Y,penalty):
 p=len(B); Y=np.asarray(Y); sd=Y.std(0); M=(Y-Y.mean(0))/np.maximum(sd,1e-8); M=M.reshape(len(I),-1); m=M.shape[1]; A=np.r_[B[:,I][:,:,I],np.eye(len(I))[None]]; a=A.reshape(p+1,-1); ref=np.r_[np.full(p,-3.),-7.]; ref[0]=-1
 def obj(q):
  v=np.exp(q); K=(v@a).reshape(len(I),-1); F=cho_factor(K); alpha=cho_solve(F,M); d=(cho_solve(F,np.eye(len(I)))-alpha@alpha.T/m).ravel(); return np.log(F[0].diagonal()).sum()+np.sum(M*alpha)/2/m+penalty*np.sum((q-ref)**2)/2,(a@d)*v/2+penalty*(q-ref)
 O=minimize(obj,ref,jac=True,method='L-BFGS-B',bounds=[(-14,5)]*p+[(-12,-2)],options={'maxiter':160,'ftol':1e-7}); v=np.exp(O.x); F=cho_factor((v@a).reshape(len(I),-1)); P=np.tensordot(v[:-1],B[:,J][:,:,I],axes=1)@cho_solve(F,M); return P*sd+Y.mean(0),O.fun
if __name__=='__main__':
 for variant in (0,1):
  for p in [.02,.15,.6,2.]:
   es=[]; et=[]; ws=[]
   for S in splits:
    I,J=S[:52],S[52:]; B=basis(I,variant); Z=[infer(H,I,J,y[I,None],p) for H in B]; P=np.array([q[0][:,0] for q in Z]); L=np.array([q[1] for q in Z]); w=np.exp(-.4*(L-L.min())); w/=w.sum(); es.append(np.r_[np.mean(abs(P-y[J]),axis=1),np.mean(abs(w@P-y[J])),np.mean(abs(P.mean(0)-y[J]))]); ws.append(w)
   e=np.array(es); print(variant,p,'first',e[0].round(5),'mean',e.mean(0).round(5),'worst',e.max(0).round(5),'weights',ws[0].round(3),flush=True)
