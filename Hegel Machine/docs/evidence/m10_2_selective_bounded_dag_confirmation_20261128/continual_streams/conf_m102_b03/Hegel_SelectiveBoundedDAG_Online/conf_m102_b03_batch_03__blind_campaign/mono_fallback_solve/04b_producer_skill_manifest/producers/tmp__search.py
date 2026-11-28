from studyfit import *
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
p=s['panel_b'];E=p['v']-one(p['u'])@ls(one(p['u']),p['v']);print('B affine rmse',np.std(E))
f=np.load('/tmp/features.npz'); z,L,C,D,Y,H=(f[k] for k in ('z','L','C','D','Y','H'));Y=Y[:,:9];
for nm,W in [('z',z),('L',L),('C',C),('D',D)]:print(nm,'range',W.min(0),W.max(0),'SD',W.std(0))

def col(A,B):return (A[:,:,None]*B[:,None,:]).reshape(len(A),-1)
def norm(A):return (A-A[:52].mean(0))/np.maximum(A[:52].std(0),1e-9)
M={'z':z,'L':L,'C':C,'D':D,'CD':np.c_[C,D],'LCD':np.c_[L,C,D],'plus':C+D,'minus':C-D,'dot':np.sum(C*D,axis=1)[:,None]};
M.update({k:np.c_[M[k],pair(M[k])] for k in ('plus','minus')}); M['bilin']=np.c_[C,D,col(C,D)]; M['diag']=np.c_[C,D,C*D];M['LC']=np.c_[L,C,col(L,C)];M['LD']=np.c_[L,D,col(L,D)];M['allpair']=np.c_[L,C,D,col(L,C),col(L,D),col(C,D),pair(L),pair(C),pair(D)];M['CDquad']=np.c_[C,D,pair(C),pair(D),col(C,D)]; M['zquad']=np.c_[z,pair(z)];M['Lquad']=np.c_[L,pair(L)];
Kp={k:1+norm(F)@norm(F).T/F.shape[1] for k,F in M.items()};
P={};
for k in ('z','L','C','D','CD','LCD','plus','minus','bilin','LC','LD','allpair'):
 A=norm(M[k]);sq=cdist(A,A,'sqeuclidean')/A.shape[1];
 for j in [.25,.5,1,2,4,8]:
  T=sq/j**2; R=np.sqrt(5*T);P[k,j,'g']=np.exp(-T);P[k,j,'q']=(1+T)**-2;P[k,j,'m']=(1+R+R*R/3)*np.exp(-R)
e=[];i=np.eye(52)
def bench(K,name,grid):
 for q in grid:
  F=cho_factor(K[:52,:52]+q*i,check_finite=False);b=cho_solve(F,Y[:52],check_finite=False);inv=cho_solve(F,i,check_finite=False).diagonal()[:,None]; cv=abs(b/inv).mean(0); y=K[52:,:52]@b;va=abs(y-Y[52:]).mean(0);e.append((name,q,cv[0],cv[1:].mean(),va[0],va[1:].mean(),y))
for k,K in Kp.items():bench(K,k,[1e-7,.0001,.002,.03])
for key,K in P.items():
 for k in ('z','CD','bilin','diag','allpair','CDquad','zquad','LC','LD','plus'):
  for w in [.1,1,10]:bench(K+w*Kp[k],(key,k,w),[1e-6,.0001,.001,.01])
for ii in [2,3,4,5]:
 print('Best by',ii)
 for a in sorted(e,key=lambda r:r[ii])[:8]:print(a[:6]);
 a=sorted(e,key=lambda r:r[ii])[:30]; w=np.exp(-20*(np.array([v[ii] for v in a])/a[0][ii]-1));p=np.average(np.stack([r[-1] for r in a]),axis=0,weights=w);print('ensemble',abs(p[:,0]-Y[52:,0]).mean(),abs(p[:,1:]-Y[52:,1:]).mean())
