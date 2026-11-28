exec(open('/tmp/hegel_explore.py').read().split('a=S[\'panel_a\']')[0]);
from scipy.spatial.distance import cdist
from scipy.linalg import solve
z,b=arch['_coordinates'](S,x); f=erf(1.43566704*z)-1.51459922*z
p=S['panel_c']; c1=D(np.c_[z,P(z,2)])@fit(D(np.c_[p['u'],P(p['u'],2)]),p['v1']); c2=D(np.c_[z,z*z])@fit(D(np.c_[p['u'],p['u']**2]),p['v2'])
Y=np.c_[y,tr]; n=len(x); folds=[np.arange(52,72)]+[i for j in [1234,735] for i in np.array_split(np.random.default_rng(j).permutation(n),4)]
score=[]
def assess(name,K,regs=(1e-5,.0001,.001,.01,.05)):
 for r in regs:
  es=[]
  for te in folds:
   it=np.setdiff1d(np.arange(n),te); pred=K[np.ix_(te,it)]@solve(K[np.ix_(it,it)]+np.eye(len(it))*r,Y[it],assume_a='pos'); e=np.abs(pred-Y[te]);es.append([e[:,0].mean(),e[:,1:].mean()])
  es=np.array(es);score.append((es.mean(0).max(),name,r,es[0],es.mean(0),es.max(0)))
std=lambda X:(X-X.mean(0))/np.maximum(X.std(0),1e-12)
lin=lambda a:a@a.T/a.shape[1]
Kz,Kf,Kb,Kc,Kd=[lin(std(a)) for a in (z,f,b,c1,c2)]
for nm,V in [('z',z),('b',b),('f',f),('c1',c1),('c2',c2)]:
 A=D(V); print(nm,'linear full train rmse',np.sqrt(np.mean((Y-A@fit(A,Y))**2,axis=0))[[0,-1]])
for kl in [.03,.15,.4,1,2.5,7,30]:
 for k in [0,.02,.1,.4]:
  for i in [2,3,4,5]:
   for intr in [0,1]:
    M=1+Kz*kl; K=M**(1+intr)+k*(Kc+Kd)+(1+Kb)**i
    if intr: K+=kl*Kz*(Kb+.5*Kb**2)
    assess(('poly',kl,k,i,intr),K)
for bw in [.15,.35,.6,1,1.8,3.5]:
 for v in [z,b,f]:
  V=std(v); Q=((V[:,None,:]-V[None,:,:])*bw)**2; R=np.exp(-Q/2); A=R.mean(2); H=2/3*(A*A-(R*R).sum(2)/16)
  for p in [0,1,2]:
   for aa in [.1,.5,2]:
    for ww in [.05,.3,1,3]:
     K=(1+Kb)**3+ww*Kz+aa*(A+p*H+(p==2)*np.exp(-Q.sum(2)/2));assess(('rbf',bw, 'z' if v is z else 'b' if v is b else 'f',p,aa,ww),K)
for i in [0,4,5]:
 print('ranking',i); print(*sorted(score,key=lambda r:r[0] if i==0 else max(r[i]))[:10],sep='\n')
print('split metrics for previous archived kernel:'); A=std(np.c_[z,b,np.tanh(z),np.tanh(2.5*b)]); assess('archive',arch['_kernel'](A,A),(.004,));print(score[-1])
