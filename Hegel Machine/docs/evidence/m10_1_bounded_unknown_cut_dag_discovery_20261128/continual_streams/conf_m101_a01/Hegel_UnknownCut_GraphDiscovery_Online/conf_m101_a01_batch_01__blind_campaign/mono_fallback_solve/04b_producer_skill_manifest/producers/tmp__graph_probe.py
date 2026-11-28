import pickle, numpy as np
from itertools import combinations_with_replacement, combinations
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; u,sv,v=np.linalg.svd(tr,full_matrices=False); z=tr@v[0]*v[0,-1]; Y=np.c_[y,z]
def affine(p,k='v'):
 d=np.c_[np.ones(len(p['u'])),p['u']]; v=p[k]; keep=np.ones(len(d),bool)
 for _ in range(5):
  f=np.linalg.lstsq(d[keep],v[keep],rcond=None)[0]; r=np.linalg.norm(v-d@f,axis=1); keep=r<max(.001,3*np.median(r))
 return f
A=affine(s['panel_a']); B=affine(s['panel_b']); C=affine(s['panel_c'],'v1')
def forward(q,f):return q@f[1:]+f[0]
def reverse(q,f):return (q-f[0])@np.linalg.inv(f[1:])
def frames(x):
 o={}
 for name,q in [('x',x),('a-',reverse(x,A)),('a+',forward(x,A))]:
  o[name]=q
  for nm,M in [('b',B),('c',C)]:
   for sig,fn in [('+',forward),('-',reverse)]:o[name+nm+sig]=fn(q,M)
 return o
def poly(x,degree):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for d in range(1,degree+1) for j in combinations_with_replacement(range(4),d)])
def kernel(q,ell,kind):
 r=np.linalg.norm(q[:,None,:]-q[None,:,:],axis=2)/ell
 if kind=='rbf':return np.exp(-r*r/2)
 r*=np.sqrt(5);return (1+r+r*r/3)*np.exp(-r)
I=np.arange(72); splits=[(I[:52],I[52:])]; rng=np.random.default_rng(36)
for i in range(5):
 p=rng.permutation(I); splits.append((p[:52],p[52:]))
def scores(K,D=None):
 out=[]
 for a,b in splits:
  L=K[np.ix_(a,a)]; R=K[np.ix_(b,a)]; co=solve(L,Y[a] if D is None else np.c_[Y[a],D[a]],assume_a='pos'); g=co[:,:2]
  if D is not None:
   w=co[:,2:]; beta=np.linalg.lstsq(D[a].T@w,D[a].T@g,rcond=1e-10)[0]; pred=R@(g-w@beta)+D[b]@beta
  else:pred=R@g
  out.append(np.mean(abs(pred-Y[b]),axis=0))
 return np.array(out)
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=150)
 P=poly(x,2); fit=np.linalg.lstsq(P,Y,rcond=None)[0]; H=np.zeros((2,4,4));
 for (i,j),v0 in zip(combinations_with_replacement(range(4),2),fit[5:]):H[:,i,j]=H[:,j,i]=v0 if i==j else v0/2
 print('quad Hess eigensystems',np.linalg.eigh(H)); print('y extreme samples',np.c_[x,y,z][np.argsort(y)[np.r_[0:7,-5:0]]]);
 for nm,q in frames(x).items():
  ca=np.linalg.lstsq(poly(q,2),Y,rcond=None)[0]; print(nm,'quad linear',ca[1:5,0], 'square',ca[[5,9,12,14],0]);
  candidates=[]; D=poly(q,2)
  for ell in [.45,.85,1.4,2.4,4.]:
   for kind in ['rbf','m52']:
    K=kernel(q,ell,kind)
    for n in [1e-5,.0003,.003,.025]:
     for mode in ['iso','add','mix']:
      if mode=='iso': Q=K
      else:
       delta=(q[:,None,:]-q[None,:,:])/ell
       Ka=np.exp(-delta**2/2).sum(axis=-1)/4
       Q=Ka+(K*.25 if mode=='mix' else 0)
      val=scores(Q+n*np.eye(72),D); candidates.append((val,ell,kind,n,mode))
  for i in range(2):
   best=min(candidates,key=lambda r:.35*r[0][0,i]+.65*np.max(r[0][1:,i])); v0,*cfg=best; print(' target',i,cfg,'required / mean random / worst random',v0[0,i],v0[1:,i].mean(),v0[1:,i].max())
