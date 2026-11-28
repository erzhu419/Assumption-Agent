from features import *
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky,solve_triangular
from scipy.optimize import minimize
import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 ns={}; exec(open('/tmp/compose.py').read().split("print('temporal residual'")[0],ns); c,d=ns['c'],ns['d']
F=pol(x,2); print('quadratic y eigensystem:',flush=True)
q=ls(F[:52],y[:52]); Q=np.zeros((4,4));
for z,(a,b) in zip(q[5:],combinations_with_replacement(range(4),2)):Q[a,b]=Q[b,a]=z if a==b else z/2
print(np.linalg.eigh(Q), 'lin',q[1:5],flush=True)
U={};
for k in ['x','a','ai','xb','ab','aib']:
 w=coords[k]; U[k]=w; U[k+'c']=c(w);U[k+'d']=d(w);U[k+'dc']=c(d(w));U[k+'cd']=d(c(w));

def kernels():
 K={}
 def add(n,G):
  G=G-F@ls(F,G); z=np.mean(G**2)*G.shape[1];
  if z>1e-16: K[n]=G@G.T/z
 for nm,W in U.items():
  W=W/np.std(W,axis=0)
  S=cdist(W,W,'sqeuclidean'); dot=W@W.T/4
  for L in [.7,1.5,3.]:
   R=np.sqrt(5*S)/L;K[nm+'m'+str(L)]=(1+R+R**2/3)*np.exp(-R)
   if nm=='x':K[nm+'e'+str(L)]=np.exp(-.5*S/L**2)
  add(nm+'nonlin',np.c_[W,np.sin(2*W),np.cos(2*W),np.tanh(2*W)])
  add(nm+'cross',np.einsum('ni,nj->nij',W,x).reshape(-1,16))
  for j in [3,4]: K[nm+'p'+str(j)]=dot**j
 # Axis and quadratic-model projection kernels complement isotropic 4D distances.
 for m in [1.,2.,4.]:
  R=np.abs(x[:,None,:]-x[None,:,:])*np.sqrt(5)/m
  d0=(1+R+R**2/3)*np.exp(-R); K['add'+str(m)]=d0.mean(2);K['pairs'+str(m)]=np.mean([d0[:,:,a]*d0[:,:,b] for a,b in combinations_with_replacement(range(4),2) if a<b],axis=0)
 return K
K=kernels();
def pred(i,j,kk,target,mean=F):
 a=mean[i]; b=mean[j]; A=kk[np.ix_(i,i)]; L=cholesky(A,lower=True); wt=solve_triangular(L,np.c_[a,target[i]],lower=True); c0=ls(wt[:,:-1],wt[:,-1]); return b@c0+kk[np.ix_(j,i)]@solve_triangular(L.T,wt[:,-1]-wt[:,:-1]@c0,lower=False)
def metric(k,target,mean=F):
 e=[np.mean(abs(pred(i,j,k,target,mean)-target[j])) for i,j in splits];return np.array([e[0],np.median(e[1:]),max(e[1:])])
for t in range(2):
 yy=Y[:,t]; tbl=[]
 for name,k in K.items():
  for rr in [1e-4,.003,.03,.3]:
   a=metric(k+rr*np.eye(n),yy);tbl.append((a,name,rr))
 print('target',t,flush=True)
 for a,nm,r in sorted(tbl,key=lambda a:a[0][0]+a[0][1]+.3*a[0][2])[:14]:print(nm,r,a,flush=True)
 for names in [['xm1.5','xm3.0','add1.0','pairs1.0','xp3','xp4'],['xm1.5','xm3.0','add1.0','pairs1.0','xp3','xp4','adcm1.5','adcp3','abcm1.5','abcp3','abcdm1.5','abcdp3'],list(K)[::4]]:
  ks=np.array([K[a] for a in names]+[np.eye(n)]); errs=[]; weights=[]
  for i,j in splits:
   q,r=np.linalg.qr(F[i],mode='complete'); N=q[:,F.shape[1]:]; yy0=N.T@yy[i]; Ks=N.T@ks[:,i][:,:,i]@N
   def loss(p):
    w=np.exp(p); C=np.einsum('k,kij->ij',w,Ks); LL=cholesky(C,lower=True); l=solve_triangular(LL,yy0,lower=True); ci=solve_triangular(LL,np.eye(len(i)-F.shape[1]),lower=True); alpha=ci.T@l
    return np.log(np.diag(LL)).sum()+.5*l@l, .5*w*np.einsum('ij,kij->k',ci.T@ci-np.outer(alpha,alpha),Ks)
   p=minimize(loss,np.r_[np.full(len(names),-4.),-8.],jac=True,bounds=[(-18,4)]*len(ks),method='L-BFGS-B',options={'maxiter':120}).x;w=np.exp(p);e=abs(pred(i,j,np.einsum('k,kij->ij',w,ks),yy)-yy[j]);errs.append(np.mean(e));weights.append(w)
  print('learned',len(names),errs[0],np.median(errs[1:]),max(errs[1:]),sorted(zip(np.round(weights[0],5),names+['noise']),reverse=True)[:8],flush=True)
