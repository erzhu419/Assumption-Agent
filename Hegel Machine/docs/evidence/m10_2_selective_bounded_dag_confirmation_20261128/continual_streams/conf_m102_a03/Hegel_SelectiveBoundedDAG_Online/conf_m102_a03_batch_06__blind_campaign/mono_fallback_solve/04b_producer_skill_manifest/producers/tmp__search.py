import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from readout import *
from scipy.optimize import minimize_scalar
from scipy.special import expit
D={'b':b,'z':z,'c':c,'d':d,'p':p,'q':q}; E={n:pol(w,2) for n,w in D.items()}; T=np.c_[y,t@V[0],t@V[1],t@V[2]]
Q=E['b']; print('b quad',fit(Q[:52],T[:52]).T)
for fn in [np.sin,np.tanh,np.cos,lambda x:x/(1+x*x),lambda x:1/(1+x*x)]:
 for j in [0,1]:
  vv=T[:,j]; aa=pol(b,2); res=vv-aa@fit(aa[:52],vv[:52]); out=[]
  for n,w in D.items():
   for h in [.5,1.,1.25,1.75,2.5,3.75,5,7.5,10]:
    for src in [w, np.c_[w,w+np.roll(w,1,axis=1),w-np.roll(w,1,axis=1),w*np.roll(w,1,axis=1)]]:
     A=np.c_[aa,fn(h*src)]; pr=A@fit(A[:52],vv[:52]); out.append((np.mean(abs(pr[52:]-vv[52:])),n,h,src.shape[1],np.std((pr-vv)[:52])))
  print(getattr(fn,'__name__'),j,sorted(out)[:4])
# Compact cross-branch and denominator alternatives; holdout ranking inspected separately.
for l in [0.,.0001,.001,.01,.1]:
 rr=[]
 for n,w in D.items():
  for m,g in D.items():
   for G in [np.c_[Q,w],np.c_[Q,w,w*g],np.c_[Q,w,w*np.roll(g,1,axis=1)], np.c_[Q,w,w[:, :, None].__mul__(g[:,None,:]).reshape(-1,16)],np.c_[Q,w,w**3],np.c_[Q,w,w**3,w**4],np.c_[Q,w,(w[:,:,None]*g[:,None,:]).reshape(-1,16), pol(w,2)[:,5:]]]:
    A=np.c_[np.ones(len(G)),norm(G[:,1:])]; co=np.linalg.solve(A[:52].T@A[:52]+np.diag(np.r_[0,np.broadcast_to(l,A.shape[1]-1)])+1e-10*np.eye(A.shape[1]),A[:52].T@T[:52]); rr.append((n,m,G.shape[1],errors(A[52:]@co),np.std((A@co-T)[:52],0)))
 for k in [0,5]: print('cross',l,k,*sorted(rr,key=lambda r:r[-2][k])[:4],sep='\n')
# Single-index nonlinear correction to quadratic and learned rational forms.
for j in [0,1]:
 vv=T[:,j]; rr=[]
 for n,w in D.items():
  for m,g in D.items():
   for k in [0,1,2,3,4]:
    A=np.c_[Q,w]
    def pred(h,which):
     z=g@h[:4]+h[4];
     if k==0:F=np.c_[A,np.sin(z)]
     if k==1:F=np.c_[A,np.tanh(z)]
     if k==2:F=np.c_[A,np.exp(-z*z)]
     if k==3:F=np.c_[A[:,:1],A/(1+z[:,None]**2)]
     if k==4:F=np.c_[A[:,:5],A[:,5:]/(1+z[:,None]**2)]
     co=fit(F[:52],vv[:52]); return F[which]@co
    rng=np.random.default_rng(47); opts=[]
    for ini in [np.ones(5)*.2,*rng.normal(0,1,(5,5))]:
     r=least_squares(lambda h:pred(h,slice(52))-vv[:52],ini,max_nfev=130,bounds=(-12,12)); opts.append(r)
    r=min(opts,key=lambda r:r.cost); rr.append((np.mean(abs(pred(r.x,slice(52,None))-vv[52:])),np.std(r.fun),n,m,k,r.x))
 print('VP',j,*sorted(rr,key=lambda a:a[0])[:8],sep='\n'); print('VP by train',j,*sorted(rr,key=lambda a:a[1])[:8],sep='\n')
