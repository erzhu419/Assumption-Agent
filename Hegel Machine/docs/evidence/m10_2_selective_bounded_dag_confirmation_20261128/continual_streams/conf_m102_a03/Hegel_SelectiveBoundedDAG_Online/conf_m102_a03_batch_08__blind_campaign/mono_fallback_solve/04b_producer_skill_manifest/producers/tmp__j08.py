import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_b'][k] for k in ('u','v')]; D=np.load('/tmp/feat08.npz'); z,y,t=[D[k] for k in ('z','y','t')]; n=52
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-11)[0]
def q(a):return np.column_stack([a[:,i]*a[:,j] for i,j in cwr(range(4),2)])
# Identify a common scalar warp independently via alternating linear solves.
u0=u.copy(); z0=z.copy(); feats=[]
for deg in [2,3,4,5,6,7,8]:
 def bas(a):return np.stack([a**(2*k+1) for k in range(deg)],axis=-1)
 U=bas(u); w=fit(np.c_[np.ones(len(u)),u],v); c=np.zeros(deg);c[0]=1
 for _ in range(100):
  X=U@c; w=fit(np.c_[np.ones(len(u)),X],v); B=np.einsum('nid,ij->njd',U,w[1:]).reshape(v.size,deg); c=fit(B-B.reshape(len(u),4,deg).mean(0)[None,:,:].repeat(len(u),axis=0).reshape(v.size,deg),(v-v.mean(0)).ravel())
 X=U@c; w=fit(np.c_[np.ones(len(u)),X],v); print('panel',deg,np.std(np.c_[np.ones(len(u)),X]@w-v),c/c[-1],flush=True); feats.append((str(deg),bas(z)@c))
np.savez('/tmp/warp08.npz',**dict(feats))
# Train-only variable projection: tuned shared warp and polynomial readout, with shrinkage for weak terms.
yh=np.c_[y,t]; rng=np.random.default_rng(555); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(6)]; results=[]
for mode in ['sin','rat','tanh','poly','rpow','shift']:
 for grammar in [0,1,2,3,4,5]:
  for target in [0,1,2]:
   es=[]; ps=[]
   for p in splits:
    I,J=p[:52],p[52:]; Y=yh[I,:] if target==2 else yh[I,:1] if target==0 else yh[I,1:];
    def design(h):
     a,b,c=h; W={'sin':lambda:z+a*np.sin(b*z)+c*z**3,'rat':lambda:z+a*z/(1+b*z*z)+c*z**3,'tanh':lambda:z+a*np.tanh(b*z)+c*z**3,'poly':lambda:z+a*z**3+b*z**5+c*z**7,'rpow':lambda:z*(1+a*z*z)/(1+b*z*z)**c,'shift':lambda:z+a*np.sin(b*z+c)}[mode]();
     groups=[np.ones((len(z),1)),W,z,q(W),q(z),np.column_stack([W[:,i]*z[:,j] for i in range(4) for j in range(4)]),W**3,W**4]
     gs=[(0,1,3),(0,1,2,3),(0,1,2,3,4),(0,1,2,3,5),(0,1,2,3,6,7),(0,1,2,3,4,5,6,7)][grammar]
     A=np.column_stack([groups[i]/np.maximum(groups[i][I].std(0),1e-4) if i else groups[i] for i in gs]); reg=np.concatenate([np.full(groups[i].shape[1],{0:1e-8,1:.002,2:.010,3:.002,4:.045,5:.030,6:.035,7:.045}[i]) for i in gs]);return A,reg
    def obj(h, ret=False):
     A,R=design(h); M=A[I]; C=cho_solve(cho_factor(M.T@M+np.diag(R**2)),M.T@Y); return A@C if ret else np.r_[(M@C-Y).ravel(),(R[:,None]*C).ravel()]
    starts={'sin':[[-.4,2.8,0],[-.5,2.1,0]],'rat':[[-1,1,.08],[-1.2,3,-.1]],'tanh':[[-.7,1.6,0],[-.8,2,-.1]],'poly':[[1,-.2,.1]],'rpow':[[1,1,.2],[-1,2,-.2]],'shift':[[-.4,2.8,0],[-.5,2.1,0]]}[mode]; lb=[-4,.05,-4] if mode!='poly' else [-5,-5,-5]; ub=[5,8,5];
    o=min((least_squares(obj,h,bounds=(lb,ub),max_nfev=60,ftol=1e-5) for h in starts),key=lambda r:r.cost); pr=obj(o.x,True); es.append(np.mean(abs(pr[J]-Y[:1]*0-(yh[J,:] if target==2 else yh[J,:1] if target==0 else yh[J,1:])),0)); ps.append(pr); 
    if not len(es)>1: par=o.x
   e=np.array(es); k=e[:,0] if target!=1 else e.mean(1); results.append((target,k.mean(),max(k),k[0],mode,grammar,par.tolist(),float(np.mean(e[0,1:])) if target==2 else None)); np.save('/tmp/pr08_%s_%d_%d.npy'%(mode,grammar,target),ps)
for tar in range(3):
 print('RESULTS',tar,*sorted([r for r in results if r[0]==tar],key=lambda r:r[1])[:10],sep='\n')
