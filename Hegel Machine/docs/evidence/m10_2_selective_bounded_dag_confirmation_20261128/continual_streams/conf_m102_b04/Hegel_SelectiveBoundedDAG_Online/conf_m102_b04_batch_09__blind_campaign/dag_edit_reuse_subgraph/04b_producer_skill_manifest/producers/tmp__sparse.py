from screen import *
from scipy.optimize import least_squares
from itertools import combinations
U=s['panel_c']['u']; V=s['panel_c']['v2']; f=lambda q,p:np.c_[aff(q),np.sqrt(1+p*q*q)-1]; p=minimize_scalar(lambda p:np.sum((f(U,p)@fit(U,V,lambda q:f(q,p))-V)**2),bounds=(.1,3),method='bounded').x; W=fit(U,V,lambda q:f(q,p)); coords['w']=f(x,p)@W
T=s['eval_times']; fun=lambda k:-np.expm1(-T[:,None]*np.array(k)); D=fun([.4214,1.8045]); A=np.linalg.lstsq(D,t.T,rcond=None)[0].T
Q=np.hstack([coords[k] for k in ['x','raw','b','v','w','vb','wb']]); names=[k+str(j) for k in ['x','raw','b','v','w','vb','wb'] for j in range(4)]; F=[]; N=[]
def add(z,k): F.append(z);N.append(k)
for z,k in zip(Q.T,names):
 for nm,g in [('lin',lambda z:z),('sq',lambda z:z*z),('cub',lambda z:z**3),('rat',lambda z:z*z/(1+.4*z*z)),('atan',lambda z:np.arctan(2*z)),('tanh',np.tanh),('sin',lambda z:np.sin(2*z)),('cos',lambda z:np.cos(2*z)),('sat',lambda z:z/np.sqrt(1+z*z))]:add(g(z),nm+k)
for i,j in combinations(range(Q.shape[1]),2):
 a,b=Q[:,i],Q[:,j]; k=names[i]+','+names[j]; add(a*b,'prod'+k);add(a*b/(1+.5*(a*a+b*b)),'ratprod'+k);add(a*b*b,'abb'+k);add(a*a*b,'aab'+k)
 for sign in [1,-1]:
  z=a+sign*b;add(np.sin(z),f'sin{sign}'+k);add(np.tanh(z),f'tanh{sign}'+k);add(np.cos(z),f'cos{sign}'+k)
for k in ['x','b','v','w']:
 for i,j,h in combinations(range(4),3):add(np.prod(coords[k][:,[i,j,h]],1),k+str((i,j,h)))
F=np.column_stack(F)
def predict(ids,te,groups):
 base=aff(x); mu=F[ids].mean(0); sd=np.maximum(F[ids].std(0),1e-9); FF=(F-mu)/sd; preds=[]
 for z in groups.T:
  C=base.copy(); used=[]; pp=[]
  for it in range(20):
   B=C[ids]; q=np.linalg.qr(B)[0]; M=FF[ids]-q@(q.T@FF[ids]); e=z[ids]-q@(q.T@z[ids]); score=abs(M.T@e)/np.sqrt(np.sum(M*M,0)+.005); j=np.argmax(score);used.append(N[j]); C=np.c_[C,FF[:,j]];B=C[ids]; p=C[te]@np.linalg.lstsq(B,z[ids],rcond=1e-8)[0]; pp.append(p)
  preds.append(pp)
 return np.array(preds),used
Z=np.c_[y,A]; allout=[]
for te in np.array_split(np.arange(n),4)+[np.arange(52,n)]:
 ids=np.setdiff1d(np.arange(n),te); P,_=predict(ids,te,Z); print('split',te[0],np.round(np.mean(abs(P[0]-y[te]),1),3)); print('traj',np.round(np.mean(abs(np.einsum('ajt,ka->jtk',P[1:],D)-t[te]),axis=(1,2)),3))
for z in Z.T:
 _,u=predict(np.arange(n),np.arange(n),z[:,None]); print('terms',u)
