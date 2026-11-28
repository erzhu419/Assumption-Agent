exec(open('/tmp/readout_probe.py').read().split('for name,p in')[0])
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
z=np.cbrt(q); X=s['train']['x']; T=np.c_[y,t]; h=s['eval_times']; idx=np.arange(52); tv=np.arange(52,n)
# Recover approximate amplitudes of the two temporal modes.
for rate in [.55,.7,.85,1,1.15,1.3,1.5,1.75,2]:
 a=np.array([1-np.exp(-rate*h),h*np.exp(-rate*h)]).T
 err=np.linalg.norm(t-(t@np.linalg.pinv(a.T))@a.T); print('time',rate,err)

def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
cp=s['panel_c']; U=cp['u']; maps=[np.c_[aff(np.arctan(2.15*U)),U,U**3],np.c_[aff(U),np.cos(1.3*U)]]
def features(z,X):
 f1=np.cos(f*z); f2=np.sin(f*z); cx=np.cos(f*X); sx=np.sin(f*X); b=aff(f1)@co; bq=aff(np.cos(f*z**3))@co; bx=aff(cx)@co
 c1=np.c_[aff(np.arctan(2.15*z)),z,z**3]@np.linalg.lstsq(maps[0],cp['v1'],rcond=None)[0]; c2=np.c_[aff(z),np.cos(1.3*z)]@np.linalg.lstsq(maps[1],cp['v2'],rcond=None)[0]
 return dict(z=z,X=X,f1=f1,f2=f2,cx=cx,sx=sx,b=b,bq=bq,bx=bx,c1=c1,c2=c2)
F=features(z,X)
D={};
for name in ['z','X','f1','f2','cx','sx','b','bq','bx','c1','c2']:
 x=F[name];
 for m in range(1,6): D[f'{name}p{m}']=x**m
 for g in [1.,1.3,2.15,2.6,4.3,5.2]:
  for k,fn in [('cos',np.cos),('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('rat',lambda v:1/(1+v*v)),('odd',lambda v:v/(1+v*v))]: D[f'{name}{k}{g}']=fn(g*x)
for name,x in F.items():
 for name2 in ['z','f1','f2','b','c1','c2']:
  xx=np.einsum('ni,nj->nij',x,F[name2]).reshape(n,-1); D[name+'*'+name2]=xx
base=np.c_[np.ones(n),z,z**3,F['f1'],F['f2']]; Y=T/np.std(T[:52],axis=0)
G=np.column_stack(list(D.values())); rownames=[f'{k}:{j}' for k,v in D.items() for j in range(v.shape[1])]; G=(G-G[idx].mean(0))/np.maximum(G[idx].std(0),1e-10)
def score(a,tr=idx,te=tv):
 w=np.linalg.lstsq(a[tr],T[tr],rcond=1e-8)[0]; e=np.abs(a[te]@w-T[te]); return [e[:,0].mean(),e[:,1:].mean()]
for start in [base,aff(F['f1']),np.ones((n,1)),np.c_[aff(F['f1']),np.arctan(2.15*z),F['cx']]]:
 a=start.copy(); mask=np.ones(G.shape[1],bool); sel=[]
 for it in range(30):
  p=np.linalg.svd(a[idx],full_matrices=False)[0]; r=Y[idx]-p@(p.T@Y[idx]); gg=G[idx]-p@(p.T@G[idx]); norms=np.linalg.norm(gg,axis=0); cs=np.linalg.norm(gg.T@r,axis=1)/np.maximum(norms,1e-9); cs[~mask]=0; k=np.argmax(cs); mask[k]=False; sel.append(rownames[k]); a=np.c_[a,G[:,k]]
  if it in [0,1,2,3,4,5,7,9,12,15,20,29]: print('sparse',start.shape[1],it,score(a),sel[-4:])
# Separable and full kernel mixtures, with scales selected on inner folds rather than outer validation.
from scipy.spatial.distance import cdist
from scipy.linalg import solve
allsets=[('z',),('X',),('f1',),('f1','z'),('f1','f2'),('f1','cx'),('f1','z','X'),('f1','c1'),('f1','c2'),('z','c1','f1','cx'),('z','f1','b'),('b','z')]
cv=np.array_split(np.random.default_rng(77).permutation(idx),4); results=[]
for keys in allsets:
 for fac in [.35,.65,1,1.6,2.8,5.]:
  for typ in ['sep','full','mix','poly']:
   L=[]
   for k in keys:
    x=F[k]/F[k][idx].std(0); dx=((x[:,None,:]-x[None,:,:])/fac)**2
    L.append({'sep':np.exp(-.5*dx).mean(2),'full':np.exp(-.5*dx.sum(2)),'mix':.5*np.exp(-.5*dx).mean(2)+.5*np.exp(-.5*dx.sum(2)),'poly':(1+x@x.T/(4*fac))**3}[typ])
   K=np.mean(L,0)+1
   for a in [1e-5,.001,.01,.07,.25]:
    def pred(tr,te):return K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+a*np.eye(len(tr)),T[tr],assume_a='pos')
    cvm=np.mean([np.mean(abs(pred(np.setdiff1d(idx,te),te)-T[te]),axis=0) for te in cv],0); e=np.mean(abs(pred(idx,tv)-T[tv]),0); results.append((cvm[0]+cvm[1:].mean(),e[0],e[1:].mean(),keys,fac,typ,a))
print('kernel inner',*sorted(results)[:8],sep='\n'); print('kernel outer exploratory',*sorted(results,key=lambda r:r[1]+r[2])[:6],sep='\n')
