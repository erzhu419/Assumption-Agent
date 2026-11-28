import pickle,itertools,numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; tr=np.arange(52); va=np.arange(52,72)
ca,cb,cm=np.load('/tmp/panels.npy',allow_pickle=True); np.set_printoptions(precision=4,suppress=True,linewidth=160)
def one(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def capply(z,k):
 _,name,_,p,C=cm[k]; w=np.exp(p[0])*z+p[1:]; f=w/(1+abs(w)) if name=='softsign' else (np.cos(w) if name=='cos' else np.sin(w)); return one(np.c_[z,f])@C
ops={'A':lambda z:z*ca[1]+ca[0],'a':lambda z:(z-ca[0])/ca[1],'B':lambda z:one(z)@cb,'b':lambda z:(z-cb[0])@np.linalg.inv(cb[1:]),'C':lambda z:capply(z,0),'D':lambda z:capply(z,1)}
z={'x':x}
for dep in range(3):
 for name,v in list(z.items()):
  if len(name)!=dep+1:continue
  for op,f in ops.items():
   if op.swapcase()==name[-1] or op==name[-1]:continue
   z[name+op]=f(v)
H=-np.expm1(-t[:,None]*np.array([1.8,.4])); lat=ls(H,T.T).T; Y=np.c_[y,lat]; products=list(itertools.combinations(range(4),2)); Ysd=Y[tr].std(0)
def feats(v,mode):
 a=[v]
 if mode in ('p2','p3','all'):a += [v*v,np.stack([v[:,i]*v[:,j] for i,j in products],axis=1)]
 if mode in ('p3',):a += [v**3,np.stack([v[:,i]**2*v[:,j] for i in range(4) for j in range(4) if i!=j],axis=1),np.stack([np.prod(v[:,idx],axis=1) for idx in itertools.combinations(range(4),3)],axis=1)]
 if mode in ('sep','all'):a += [np.sin(v),np.cos(v),np.sin(2*v),np.cos(2*v),v/(1+abs(v)),np.tanh(v)]
 if mode=='all':
  w=np.stack([v[:,i]+j*v[:,k] for i,k in products for j in (-1,1)],axis=1); a += [np.sin(w),np.cos(w),np.tanh(w)]
 return np.column_stack(a)
def evaluate(F,alpha):
 m=F[tr].mean(0); sd=np.maximum(F[tr].std(0),1e-10); F=(F-m)/sd
 reg=Ridge(alpha=alpha).fit(F[tr],Y[tr]); P=reg.predict(F[va]); E=P-Y[va]; return np.r_[abs(E).mean(0),abs(E[:,1:]@H.T).mean()],np.sqrt(np.mean((reg.predict(F[tr])-Y[tr])**2,axis=0))
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
print('singular values',np.linalg.svd(T,compute_uv=False)); print('trajectory+linear scalar',abs(one(np.c_[T,x])[va]@ls(one(np.c_[T,x])[tr],y[tr])-y[va]).mean())
Y=np.c_[y,T]; ix=np.arange(len(x)); n=len(x)
def kernels(z,keys,kind):
 K=[]; nm=[]
 def add(k,M):
  K.append(M/max(np.mean(np.diag(M)),1e-12)); nm.append(k)
 def radial(D):return np.exp(-D*.5) if kind in (0,2) else (1+np.sqrt(5*D)+5*D/3)*np.exp(-np.sqrt(5*D))
 for k in keys:
  v=z[k]; v=(v-v.mean(0))/np.maximum(v.std(0),1e-10); d=cdist(v,v,'sqeuclidean'); d1=(v[:,None,:]-v[None,:,:])**2; dot=v@v.T/4
  for l in [.45,.8,1.4,2.4,4.0]:
   add((k,'rbf',l),radial(d/l**2))
   if kind<2:
    add((k,'add',l),radial(d1/l**2).mean(2)); add((k,'pair',l),sum(radial((d1[:,:,i]+d1[:,:,j])/l**2) for i,j in products)/6)
  if kind!=2:
   for p in [2,3]:add((k,'poly',p),(1+dot)**p)
 return np.asarray(K),nm

def fit(K,yy,ids,others,pen=.02,objective='ml'):
 m=yy[ids].mean(); sd=yy[ids].std(); q=(yy[ids]-m)/sd; L=len(ids); I=np.eye(L); V=K[:,ids][:,:,ids].copy(); J=np.r_[V,I[None,:,:]]; diag=np.diag_indices(L)
 def fn(h):
  w=np.exp(h); A=np.einsum('i,ijk->jk',w,J); F=cho_factor(A,check_finite=False); a=cho_solve(F,q,check_finite=False); C=cho_solve(F,I,check_finite=False)
  if objective=='ml':
   val=q@a/2+np.log(F[0].diagonal()).sum(); G=(C-np.outer(a,a))/2
  else:
   D=C[diag]; e=a/D; val=np.mean(e*e)*L/2; G=-np.outer(C@(e/D),a)+ C@np.diag(e*e/D)@C
  val+=pen*sum(w[:-1]); gr=w*np.einsum('ijk,jk->i',J,G); gr[:-1]+=pen*w[:-1]; return val,gr
 h0=np.r_[np.full(len(K),-np.log(len(K))),-6.]; opt=minimize(fn,h0,jac=True,method='L-BFGS-B',bounds=[(-17,6)]*len(K)+[(-14,0)],options={'maxiter':160,'ftol':1e-8}); w=np.exp(opt.x); A=np.einsum('i,ijk->jk',w[:-1],V)+w[-1]*I; F=cho_factor(A); a=cho_solve(F,q); C=cho_solve(F,I); out=np.einsum('i,ijk,k->j',w[:-1],K[:,others][:,:,ids],a)*sd+m; return out,abs(a/np.diag(C)).mean()*sd,w
if __name__=='__main__':
 for keys in [('xa',),('xa','xaB','xaC','xaD','xaBC','xaBD'),('x','xa','xA','xaB','xaC','xaD','xaBC','xaBD','xABC','xABD','xC','xD','xaCB','xaDB','xAC','xAD')]:
  for kind in range(3):
   K,nm=kernels(z,keys,kind)
   for obj in ['ml','loo']:
    out,lo,w=fit(K,y,tr,va,objective=obj); ix=np.argsort(w[:-1])[-5:][::-1]; print(keys,kind,obj,'y',abs(out-y[va]).mean(),'loo',lo,'noise',w[-1],'weights',[(nm[i],round(w[i],3)) for i in ix],flush=True)
    if len(keys)==6:
     P=np.array([fit(K,T[:,j],tr,va,objective=obj)[0] for j in [0,3,7]]).T; print('state',abs(P-T[va][:,[0,3,7]]).mean(0),flush=True)
