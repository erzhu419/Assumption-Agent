import pickle,numpy as np
from itertools import combinations_with_replacement as comb
from scipy.optimize import minimize_scalar
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def af(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def soft(x,g):return x/np.sqrt(1+g*x*x)
def basis(x,d):return np.column_stack([x[:,j].prod(1) for k in range(d+1) for j in comb(range(x.shape[1]),k)])
a=S['panel_a']; A=ls(af(a['u']),a['v']);
for _ in range(10):
 ix=np.argsort(np.linalg.norm(af(a['u'])@A-a['v'],axis=1))[:45]; A=ls(af(a['u'][ix]),a['v'][ix])
b=S['panel_b']; B=ls(af(b['u']),b['v']); c=S['panel_c']; u=c['u'];
fun={'soft':soft,'sin':lambda x,g:np.sin(g*x),'tanh':lambda x,g:np.tanh(g*x),'atan':lambda x,g:np.arctan(g*x),'rat':lambda x,g:x/(1+g*x*x),'cos':lambda x,g:np.cos(g*x),'sqexp':lambda x,g:np.exp(-g*x*x),'erf':lambda x,g:__import__('scipy.special',fromlist=['erf']).erf(g*x),'sq':lambda x,g:x*x}
models={}
for k in ['v1','v2']:
 out=[]
 for n,f in fun.items():
  for affine in [False,True]:
   def des(x,g):return np.c_[af(x) if affine else np.ones(len(x)),f(x,g)]
   def obj(g,loo=False):
    X=des(u,g); Q=np.linalg.qr(X,mode='reduced')[0]; y=c[k]; e=y-X@ls(X,y)
    if loo:e=e/(1-np.sum(Q*Q,1))[:,None]
    return np.sqrt(np.mean(e**2))
   g=minimize_scalar(obj,bounds=(.05,7),method='bounded').x
   out.append((obj(g,True),obj(g),n,affine,g))
 print(k,sorted(out)[:8],flush=True)
 _,_,n,affine,g=min(out); f=fun[n]; coef=ls(np.c_[af(u) if affine else np.ones(len(u)),f(u,g)],c[k]); print('COEF',k,coef,flush=True)
 models[k]=(f,affine,g,coef)
def panel(x,k):
 f,affine,g,coef=models[k];return np.c_[af(x) if affine else np.ones(len(x)),f(x,g)]@coef
x=S['train']['x']; y=S['train']['y']; T=S['train']['traj']; t=S['eval_times']; Y=np.c_[y,T]; splits=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(531)
for i in range(7):
 p=rng.permutation(72); splits.append((p[:52],p[52:]))
Z={}
for name,w in [('r',x),('a',(x-A[0])/A[1:].diagonal()),('f',af(x)@A)]:
 zb=af(w)@B; Z[name]=w; Z[name+'b']=zb
 for key,inp in [('',w),('b',zb),('bs',soft(zb,.22)),('s',soft(w,.22)),('t',np.tanh(zb))]:
  for k in ['v1','v2']:Z[name+key+k]=panel(inp,k)
for name in ['r','a','f']:
 for k in ['v1','v2']: Z[name+k+'b']=af(Z[name+k])@B

def score(F,Y,pen):
 K=F@F.T; E=[]; tr=[]
 for i,j in splits:
  A=K[np.ix_(i,i)].copy(); A.flat[::len(i)+1]+=pen; a=np.linalg.solve(A,Y[i]); yh=K[np.ix_(j,i)]@a; e=abs(yh-Y[j]); E.append([e[:,0].mean(),e[:,1:].mean()]);tr.append(np.sqrt(np.mean((pen*a)**2,axis=0)))
 E=np.array(E); return np.r_[E[0],np.mean(E[1:],0),np.max(E[1:],0),tr[0][0], np.mean(tr[0][1:])]

def feats(w,n):
 for g in [0,.09,.22,.5,1,2,4]:
  for f in ([soft] if not g else [soft,np.sin if False else fun['tanh']]):
   yield (g,f.__name__),basis(f(w,g),n)
if __name__=='__main__':
 R=[]
 for name,w in Z.items():
  for d in (1,2,3,4):
   for (g,f),P in feats(w,d):
    D=np.maximum(P[:52].std(0),.1); D[0]=1
    for norm in [0,1]:
     F=P if not norm else P/D
     for a in [.00001,.001,.02,.15]:
      R.append((score(F,Y,a),(name,d,g,f,norm,a)))
 for i in [0,1,2,3,6,7]:
  print('RANK',i,flush=True)
  for z,l in sorted(R,key=lambda r:r[0][i])[:11]:print(l,np.round(z,5))
