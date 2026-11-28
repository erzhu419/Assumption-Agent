exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cw
from scipy.linalg import qr
fit=lambda a,y:np.linalg.lstsq(a,y,rcond=1e-9)[0]
cb=lambda q:np.c_[np.ones(len(q)),np.cos(2.4*q)]
c=(cb(x)@fit(cb(u),p['v2']))[:n]; x=x[:n];b=b[:n];Y=np.c_[y,tr]; k=52
U=s['panel_b']['u']; V=s['panel_b']['v']; print('panel b ranges',U.min(),U.max(),V.min(),V.max());print('panel b corr',np.round(np.corrcoef(np.c_[U,V].T)[:4,4:],2));print('panel c1 alignment',np.mean(abs(np.c_[np.ones(len(U)),*[U[:,i]*U[:,j] for i in range(4) for j in range(i+1,4)]]@fit(np.c_[np.ones(len(u)),*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]],p['v1'])-V)))
def ev(A):
 A=np.c_[np.ones(n),A]; P=A@fit(A[:k],Y[:k]); e=np.abs(P-Y); return np.round([e[:k,0].mean(),e[k:,0].mean(),e[:k,1:].mean(),e[k:,1:].mean()],5)
def basis(z,typ,g=1):
 p=z.T; su=np.array([p[i]+p[j] for i in range(4) for j in range(i+1,4)]).T; di=np.array([p[i]-p[j] for i in range(4) for j in range(i+1,4)]).T; pr=np.array([p[i]*p[j] for i in range(4) for j in range(i+1,4)]).T
 f=getattr(np,typ.split(':')[0]) if ':' in typ else None
 opts={'p1':z,'p2':np.c_[z,z*z,pr], 'v1':pr,'cyc+':z+np.roll(z,-1,axis=1),'cyc-':z-np.roll(z,-1,axis=1),'self':z,'sum':su,'diff':di,'prod':pr,'s+d':np.c_[su,di]}
 A=opts[typ.split(':')[-1]];return np.c_[np.ones(len(z)),f(g*A) if f else A]
for typ in ('p1','p2','v1',*[f'{f}:{t}' for f in ('sin','cos','tanh') for t in ('cyc+','cyc-','self','sum','diff','prod','s+d')]):
 best=None
 for g in (.5,.7,1,1.2,1.5,1.8,2,2.4,3):
  A=basis(U,typ,g);w=fit(A,V); h=np.sum(A*np.linalg.pinv(A).T,axis=1);e=np.mean(np.abs((A@w-V)/(1-h[:,None])));z=basis(x,typ,g)@w; v=ev(np.c_[b,c,z]); cand=(e,g,v)
  if best is None or cand[0]<best[0]:best=cand
 print(typ,best)
# Fit arbitrary affine projections inside one nonlinear term per panel output, allowing a linear residual.
for name,F in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos)]:
 for degree in (0,1,2):
  A=np.c_[np.ones(len(U)),U] if degree else np.ones((len(U),1)); AX=np.c_[np.ones(n),x] if degree else np.ones((n,1))
  if degree==2:
   A=np.c_[A,U*U,U*np.roll(U,-1,axis=1)];AX=np.c_[AX,x*x,x*np.roll(x,-1,axis=1)]
  Z=np.c_[U,np.ones(len(U))]; ZX=np.c_[x,np.ones(n)];rng=np.random.default_rng(99); out=[]; rr=[]
  for target in V.T:
   def mat(z):return np.c_[A,F(Z@z)]
   def fun(z):M=mat(z); return M@fit(M,target)-target
   r=min((least_squares(fun,rng.normal(0,1,5),max_nfev=100) for i in range(20)),key=lambda r:np.linalg.norm(r.fun));rr.append(np.mean(abs(r.fun))); out.append(np.c_[AX,F(ZX@r.x)]@fit(mat(r.x),target))
  print('pbr',name,degree,np.round(rr,5),ev(np.c_[b,c,*out]))
# How much of the readout lies in degree/symmetry structured bases after calibration?
blocks={'x':x,'b':b,'c':c,'bc':np.c_[b,c],'bcx':np.c_[b,c,x]}
for tag,Z in blocks.items():
 for d in range(1,5):
  P=np.column_stack([np.prod(Z[:,j],axis=1) for l in range(1,d+1) for j in cw(range(Z.shape[1]),l)]);print('poly',tag,d,ev(P))
# Derive the shared amplitudes and test if a small set of physical products can recover all.
T=s['eval_times']; Ut,Ss,Vt=np.linalg.svd(Y[:k],full_matrices=False); Z=Y@Vt[:4].T
for a0 in (np.c_[np.ones(n),b,c],np.c_[np.ones(n),x,b,c]):
 R=Y-a0@fit(a0[:k],Y[:k]);print('residual modes',np.round(np.linalg.svd(R[:k],full_matrices=False)[2][:4],3))
print('old cal panel structural clues');
for i in range(4):print('v1',i,np.round(fit(np.c_[np.ones(len(u)),*[u[:,a]*u[:,j] for a in range(4) for j in range(a+1,4)]],p['v1'])[:,i],3))
