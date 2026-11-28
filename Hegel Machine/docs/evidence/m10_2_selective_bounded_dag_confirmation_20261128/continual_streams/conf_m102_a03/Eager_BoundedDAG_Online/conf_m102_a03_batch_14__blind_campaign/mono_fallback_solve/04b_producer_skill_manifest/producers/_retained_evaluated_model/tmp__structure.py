from ops import *
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=145)
F=features(x);z=F['z'];q=F['zB']; Z=np.c_[y,T]; I=np.arange(52); J=np.arange(52,72)
for key,p,k,g in [('B','panel_b','v','sin'),('C','panel_c','v1','tanh'),('D','panel_c','v2','cos')]:
 u=s[p]['u'];v=s[p][k]; f=fn[g]
 def main(theta,ret=False):
  a=aff(u+theta[0]*f(u,theta[1])); c=ls(a,v);e=a@c-v;return (c,e) if ret else e.ravel()
 for ini in [[-.4,2.9],[.5,2.2],[-.6,1.25]]:
  r=least_squares(main,ini,max_nfev=70);c,e=main(r.x,True);print('structured',key,r.x,np.sqrt(np.mean(e*e)),np.linalg.svd(e,compute_uv=False))
for w in [q,z]:
 d=aff(np.c_[w,*[w[:,i]*w[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]]);c=ls(d[I],Z[I]); R=Z-d@c
 for k in [0,1,8]:
  H=np.zeros((4,4));
  for a,(i,j) in zip(c[5:,k],itertools.combinations_with_replacement(range(4),2)):
   H[i,j]=H[j,i]=a/(1+(i!=j))
  print('quad',k,c[:5,k],np.linalg.eigh(H))
 print('residual correlation',np.corrcoef(R.T)[0]);print('target sv',np.linalg.svd(Z-Z.mean(0),compute_uv=False));print('traj sv',np.linalg.svd(T-T.mean(0),compute_uv=False));print('residual sv',np.linalg.svd(R,compute_uv=False))
for withdrift in [False,True]:
 for rate in [.6,.8,1.,1.25]:
  a=(1-np.exp(-rate*times))[:,None];a=np.c_[a,times] if withdrift else a
  c=ls(a,T.T).T;print('time',withdrift,rate,'MAE',np.mean(abs(c@a.T-T)), 'amplitude range',c.min(0),c.max(0),'corr amp y',np.corrcoef(np.c_[c,y].T)[-1])
# Compare ridge-stabilized feature grammars, selecting penalties on shuffled folds inside the 52.
rng=np.random.default_rng(428);folds=np.array_split(rng.permutation(I),5)
def check(a):
 a=a.copy();sd=a[I].std(0); a=a[:,sd>1e-9];a=(a-a[I].mean(0))/sd[sd>1e-9];a=aff(a)
 vals=[]
 for l in [.0001,.003,.02,.1,.4,1.5,6]:
  def fit(ii):
   reg=np.eye(a.shape[1])*l;reg[0,0]=0;return np.linalg.solve(a[ii].T@a[ii]+reg,a[ii].T@Z[ii])
  es=np.array([np.mean(abs(a[j]@fit(np.setdiff1d(I,j))-Z[j]),axis=0) for j in folds]); e=abs(a[J]@fit(I)-Z[J]).mean(0)
  vals.append((es[:,0].mean(),es[:,1:].mean(),e[0],e[1:].mean(),l))
 return min(vals,key=lambda r:r[0]),min(vals,key=lambda r:r[1])
def pair(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
bs={k:F[k] for k in ['z','zB','zBC','zBD','zC','zD']}
for prefix in ['z','zB']:
 a=F[prefix]; b=F['zB' if prefix=='z' else 'z'];bs[prefix+'both']=np.c_[a,b];
 for g in fn:
  for w in [.65,1.25,2.15,2.9,3.6,4.5]:bs[prefix+g+str(w)]=np.c_[a,fn[g](a,w)]
res=[]
for k,a in bs.items():
 r,t=check(np.c_[a,pair(a,a)]);res.append((r,t,k))
for k in ['z','zB','zBC','zBD','zC','zD']:
 a=F[k];b=q;res.append((*check(np.c_[a,b,pair(a,a),pair(a,b),pair(b,b)]),'mix'+k))
for j in [0,1]:
 print('sorted',j);print(*sorted(res,key=lambda r:r[j][j])[:12],sep='\n')
