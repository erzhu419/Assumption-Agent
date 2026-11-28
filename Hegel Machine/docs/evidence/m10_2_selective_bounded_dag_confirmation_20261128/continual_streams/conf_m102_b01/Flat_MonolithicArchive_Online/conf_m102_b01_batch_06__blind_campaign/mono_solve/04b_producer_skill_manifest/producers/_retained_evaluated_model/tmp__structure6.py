import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from refine6 import *
from scipy.special import expit
N=len(Y); core=A(np.c_[w,q,Z]); state=A(np.c_[q,Z.sum(1)])
def quadratic(a,b=None):
 if b is not None:return np.einsum('ni,nj->nij',a,b).reshape(N,-1)
 return np.column_stack([a[:,i]*a[:,j] for i in range(a.shape[1]) for j in range(i,a.shape[1])])
S0=np.c_[core,w[:,0]*q[:,0],q[:,1]*R[:,0],q[:,0]*r[:,2],(q[:,2]-r[:,2])*R[:,2]]
S1=np.c_[S0,q[:,0]/(1+.22*w[:,2]**2)]
splits=[(np.arange(52),np.arange(52,N))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(N) for i in range(12)]]
def eval1(D,tr,te,j,al=0):
 U=D[tr];V=D[te];sd=np.maximum(U.std(0),1e-9);sd[0]=1; U=U/sd;V=V/sd
 coef=L(np.r_[U,np.sqrt(al*len(tr))*np.diag(np.r_[0,np.ones(U.shape[1]-1)])],np.r_[Y[tr][:,j],np.zeros((U.shape[1],len(j)))]) if al else L(U,Y[tr][:,j]);return np.mean(abs(V@coef-Y[te][:,j]))
def stats(name,d,j=[0],a=0):
 es=[eval1(d,tr,te,j,a) for tr,te in splits];print(name,a,np.round([es[0],np.mean(es[1:]),max(es[1:])],5))
for k,D in [('state',state),('state free',np.c_[state,Z]),('state wq',np.c_[core])]:
 stats(k,D,list(range(1,9)))
print('state-coefs',L(state[:52],Y[:52,1:]))
for k,D in [('core',core),('S0',S0),('S1',S1),('cubic',np.c_[S0,q[:,0]*w[:,2]**2]),('cubicC',np.c_[S0,q[:,0]**3,q[:,0]*w[:,2]**2]),('fullq2',np.c_[state,quadratic(q)]),('fullq3',np.c_[state,poly(q,3)[:,5:]]),('sparse12',np.c_[base[-1][1],C(q)[:,0]**2,q[:,0]**3,q[:,0]*(q[:,3]-r[:,3]),q[:,0]/(1+1.25*q[:,1]**2)])]:
 for a in [0,.0001,.001]: stats(k,D,[0],a)
# Try saturating the entire linear numerator with a single profiled denominator.
for dn,d in [('w',w),('z',z),('q',q),('r',r),('R',R),('state',state[:,1:])]:
 for model in ('quot','addquot','sat','tanh'):
  outs=[]
  for tr,te in splits[:5]:
   def pred(p):
    h=A(d)@p
    if model=='quot': return np.c_[core/(1+h[:,None]**2)]
    if model=='addquot': return np.c_[state,core[:,1:]/(1+h[:,None]**2)]
    if model=='sat': return np.c_[core,h/(1+h*h)]
    return np.c_[core,np.tanh(h)]
   def f(p):
    a=pred(p)[tr];return np.r_[a@L(a,Y[tr,0])-Y[tr,0],.001*p]
   rng=np.random.default_rng(41); fs=[least_squares(f,rng.normal(0,.6,d.shape[1]+1),max_nfev=150) for _ in range(7)];opt=min(fs,key=lambda f:f.cost);outs.append([eval1(pred(opt.x),tr,te,[0]),np.mean(abs(opt.fun[:len(tr)]))]);
   if len(outs)==1:par=opt.x
  print('profile',dn,model,np.round(outs,5).tolist(),np.round(par,4))
