from model import *
from scipy.linalg import qr
tr=s['train']; X=tr['x'];Y=tr['y'];T=tr['traj']; times=s['eval_times'];
def cal(x):
 def fit(k,coeff=False):
  mats=[np.c_[np.ones(len(u)),u[:,j],np.sin(k*u[:,j]),np.sin(k*u[:,(j+2)%4])] for j in range(4)]; w=[ls(d,v[:,j]) for j,d in enumerate(mats)];return np.array(w) if coeff else (np.column_stack([d@z for d,z in zip(mats,w)])-v).ravel()
 k=least_squares(fit,[2.4]).x[0];w=fit(k,True);print('cal',k,rm(fit(k)),w)
 b,a,p,q=w.T; i=(np.arange(4)+2)%4; z=(x-b)/a
 for _ in range(25):
  r=b+a*z+p*np.sin(k*z)+q*np.sin(k*z[:,i])-x; jac=np.zeros((len(z),4,4));j=np.arange(4);jac[:,j,j]=a+p*k*np.cos(k*z);jac[:,j,i]=q*k*np.cos(k*z[:,i]);z-=np.linalg.solve(jac,r[...,None])[...,0]
 return z
x=cal(X); b=s['panel_b']; M=ls(af(b['u']),b['v']); bx=af(x)@M
cb=lambda x,k:np.c_[af(x),x/(1+k*abs(x))]; C=ls(cb(c,.83),v1); Q=ls(np.c_[af(c),c*c],v2)
F=lambda p:-np.expm1(-times[:,None]*p);p=least_squares(lambda p:(F(p)@ls(F(p),T[:52].T)-T[:52].T).ravel(),[.25,2.2],bounds=(.001,10)).x; f=F(p);print('time',times,p,rm(T-ls(f,T.T).T@f.T)); ft=qr(f,mode='economic')[0]; Z=np.c_[Y,T@ft]; sc=Z[:52].std(axis=0); Z/=sc

def assess(d,w,ii,jj):
 er=(d[jj]@w-Z[jj])*sc;return np.array([mae(er[:,0]),mae(er[:,1:]@ft.T).mean()])

def library(h,mode):
 pairs=list(combinations(range(h.shape[1]),2)); a,b,c,d=h.T
 H=[h[:,i] for i in range(4)]; ns=[str(i) for i in range(4)];
 for i,j in pairs:
  for k in [1,-1,.5,-.5,2,-2]: H.append(h[:,i]+k*h[:,j]);ns.append(f'{i}+{k}*{j}')
 H=np.array(H).T; out=[]; names=[]
 def add(M,N):out.extend(M.T);names.extend(N)
 add(np.c_[h,h*h,*[h[:,i]*h[:,j] for i,j in pairs]],['linear'+n for n in ns[:4]]+['square'+n for n in ns[:4]]+[f'{i}*{j}' for i,j in pairs]);
 add(np.c_[h**3,*[np.prod(h[:,idx],axis=1) for idx in combinations_with_replacement(range(4),3) if len(set(idx))>1]],['cube'+i for i in ns[:4]]+[str(idx) for idx in combinations_with_replacement(range(4),3) if len(set(idx))>1]);
 for name,fn in funcs().items():
  if name=='sinh':continue
  for k in ([.5,1.,2.,3.] if name in ('sin','tanh','atan') else [.22,.5,.85,1.5,3.]):
   add(fn(H,k),[f'{name}:{k}:{i}' for i in ns]);
 for k in [.22,.5,1.]:
  for j in range(4):
   for power in [1,2]: add(h**power/(1+k*h[:,j:j+1]**2),[f'{i}^{power}/(1+{k}*{j}^2)' for i in range(4)])
 if mode: 
  add(np.c_[cb(h,.83)@C,np.c_[af(h),h*h]@Q],[f'C{i}' for i in range(4)]+[f'Q{i}' for i in range(4)])
 return np.column_stack(out),names

def greedy(base,D,names,ii,jj, n=20):
 A=base.copy();Q=qr(A[ii],mode='economic')[0]; R=D[ii]-Q@(Q.T@D[ii]); z=Z[ii]-Q@(Q.T@Z[ii]); taken=[]
 for s in range(n):
  norm=np.linalg.norm(R,axis=0); cor=R.T@z; idx=np.argmax(np.linalg.norm(cor,axis=1)/np.maximum(norm,1e-8));taken.append(names[idx]);col=R[:,idx]/max(norm[idx],1e-9);R-=np.outer(col,col@R);z-=np.outer(col,col@z);A=np.c_[A,D[:,idx]]
  if s in [1,3,5,7,9,13,17,19]:
   w=ls(A[ii],Z[ii]);print(s+1, np.round(assess(A,w,ii,ii),4),np.round(assess(A,w,ii,jj),4),taken[-4:])
 return A,taken
if __name__=='__main__':
 for n,h in [('x',x),('B',bx),('C',cb(x,.83)@C),('C(B)',cb(bx,.83)@C)]:
  print(n);d,nm=library(h,1);a,taken=greedy(af(h),d,nm,np.arange(52),np.arange(52,72));print(taken)
