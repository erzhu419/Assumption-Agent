from explore import *
from scipy.special import erf
from itertools import combinations_with_replacement
B=bv; U=u; C=np.sum(U*np.roll(B,1,axis=1),axis=1); n=len(y); ix=np.arange(n)
W=np.column_stack((np.ones(n), B, C)); Q=np.column_stack([B[:,i]*B[:,j] for i,j in combinations_with_replacement(range(4),2)])
rng=np.random.default_rng(751); folds=[(ix[:52],ix[52:])]+[(p[:52],p[52:]) for p in (rng.permutation(n) for _ in range(24))]
for z in [*U.T,abs(C),np.linalg.norm(U,axis=1),np.linalg.norm(B,axis=1)]:
 j=np.argsort(z); folds.append((j[:-15],j[-15:]))

def extract(p):
 A=np.zeros((4,4))
 for m,(i,j) in enumerate(combinations_with_replacement(range(4),2)): A[i,j]=A[j,i]=p[m]/(1+(i!=j))
 return np.linalg.eigh(A)

def pred(p,I,mode):
 b=B[I]; c=C[I]; x=b@p[1:5]+p[5]; v=p[0]*np.tanh(x)+p[6]*c+p[7]; q=p[8:]
 if mode=='quad': return v+Q[I]@q
 if mode=='same2': return v+q[0]*x*x
 if mode=='same2b': return v+q[0]*np.tanh(x)**2
 if mode=='rank1': return v+q[0]*(b@q[1:])**2
 if mode=='rank1odd': return v+q[0]*np.tanh(b@q[1:])**2
 if mode=='diag': return v+(b*b)@q
 if mode=='crossx': return v+q[0]*c*x
 if mode=='crossx2': return v+q[0]*c*x*x
 if mode=='crossden': return v-p[6]*c*(q[0]*x)**2/(1+(q[0]*x)**2)
 if mode=='bden': return v-p[0]*np.tanh(x)*(b@q)**2/(1+(b@q)**2)
 if mode=='linbden': return p[0]*x/(1+(b@q)**2)+p[6]*c+p[7]
 if mode=='crosscden': return v-p[6]*c*(q[0]*c)**2/(1+(q[0]*c)**2)
 return v
modes={'base':[],'quad':[0.]*10,'same2':[0.],'same2b':[0.],'rank1':[1.,.2,.2,-.2,.2],'rank1odd':[1.,.2,.2,-.2,.2],'diag':[0.]*4,'crossx':[0.],'crossx2':[0.],'crossden':[.5],'bden':[.2,.2,-.2,.2],'linbden':[.2,.2,-.2,.2],'crosscden':[.3]}
for mode,q0 in modes.items():
 errs=[]
 for I,J in folds:
  p0=np.linalg.lstsq(W[I],y[I],rcond=None)[0]; initial=np.r_[2.,p0[1:5]/2,0.,p0[5],p0[0],q0]; fits=[least_squares(lambda p:pred(p,I,mode)-y[I],initial,max_nfev=220)];
  if mode.startswith('rank1'):
   for sign in [-1,1]:
    alt=initial.copy(); alt[8:]=np.r_[sign,p0[1:5]/3]; fits.append(least_squares(lambda p:pred(p,I,mode)-y[I],alt,max_nfev=220))
  fit=min(fits,key=lambda f:f.cost); errs.append(np.mean(abs(pred(fit.x,J,mode)-y[J])));
  if len(errs)==1:
   fit0=np.mean(abs(fit.fun)); detail=np.round(fit.x[8:],3)
   if mode=='quad':
    eig,vec=extract(fit.x[8:]); print('quad eig',np.round(eig,4),'eigenvectors',np.round(vec,3))
 print(mode,np.round([errs[0],np.mean(errs[:25]),max(errs[:25]),max(errs[25:]),fit0],6),detail)
for name,F in [('base',W),('base+u',np.c_[W,U]),('base+quadB',np.c_[W,Q])]:
 errors=[]
 for I,J in folds:
  E=F[J]@np.linalg.lstsq(F[I],tr[I],rcond=None)[0]-tr[J]; errors.append(np.mean(abs(E),axis=0))
 E=tr-F@np.linalg.lstsq(F,tr,rcond=None)[0]; X,V,Y=np.linalg.svd(E,full_matrices=False); q0=X[:,0]*V[0]; print('trajectory',name,np.round([np.mean(errors[0]),np.max(np.mean(errors[:25],axis=1)),np.max(np.mean(errors[25:],axis=1))],6),'resid sv',np.round(V[:4],5),'shape',np.round(Y[0],4),'resid quad vector',np.round(np.linalg.lstsq(np.c_[W,Q],q0,rcond=None)[0][6:],3))
