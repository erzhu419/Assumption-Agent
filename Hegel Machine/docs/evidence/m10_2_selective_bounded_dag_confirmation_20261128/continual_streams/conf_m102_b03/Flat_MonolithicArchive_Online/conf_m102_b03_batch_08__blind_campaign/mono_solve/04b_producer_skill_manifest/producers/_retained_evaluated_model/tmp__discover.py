exec(open('/tmp/graph.py').read().split("print('B'")[0])
from scipy.linalg import solve
from scipy.optimize import least_squares
tr=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; n=52
H=np.c_[-np.expm1(-1.8*t),-np.expm1(-.4*t),t/(1+3.2*t)]; Y=np.c_[y,ls(H,tr.T).T]; scales=Y[:n].std(0); Y=Y/scales
roots={'z':z,'B':ops['B'](z),'Bc':z@b[1:],'U':(z-b[0])@np.linalg.inv(b[1:]),'x':x}; F={}; scores=[]
def add(k,v):
 if v.ndim==1:v=v[:,None]
 F[k]=v
for k,v in roots.items():
 for f,w in [('raw',v),('tanh',np.tanh(1.25*v)),('sin',np.sin(1.25*v)),('bend',v+.22*adj(v)),('rat',v/(1+.22*v*v)),('warp',np.tanh(1.25*v)+.22*adj(v)),('warps',np.sin(1.25*v)+.22*adj(v))]:
  name=k+f; p=ops['P'](w);q=ops['Q'](w); bb=ops['B'](w)
  for j,a in [('p',p),('q',q),('w',w),('b',bb),('pq',p*q),('dot',(p*q).sum(1)),('p+q',p+q),('p-q',p-q),('pr',p/(1+.22*q*q)),('qr',q/(1+.22*p*p)),('bp',ops['P'](bb)),('bq',ops['Q'](bb))]:
   add(name+j,a)
   if j in ('p','q','b','p+q','p-q','bp','bq'):
    for l,c in [('t',np.tanh(1.25*a)),('s',np.sin(1.25*a)),('r',a/(1+.22*a*a)),('l',np.logaddexp(0,1.25*a)),('x',a*a),('c',np.cos(1.25*a))]:add(name+j+l,c)
def ev(M,target):
 A=D(M); h=ls(A[:n],target[:n]); e=abs(A@h-target); loo=e[:n]/np.maximum(1-np.sum(A[:n]*np.linalg.pinv(A[:n]).T,axis=1),1e-5);return (loo.mean(),e[n:].mean(),e[:n].mean())
for m in range(3):
 target=Y[:,m]; best=sorted((ev(f,target),k) for k,f in F.items()); print('SINGLE',m,best[:10],flush=True)
 keys=[k for _,k in best[:28]]; pairs=sorted((ev(np.c_[F[k],F[l]],target),k,l) for k,l in it.combinations(keys,2)); print('PAIR',m,pairs[:8],flush=True)
 for j in range(3):
  opt=[]
  for _,k,l in pairs[:8]:
   v=np.c_[F[k],F[l]] if isinstance(l,str) else np.c_[F[k],*[F[a] for a in l]]
   for c,w in F.items():
    if c==k or c in l:continue
    opt.append((ev(np.c_[v,w],target),k,(l,c) if isinstance(l,str) else (*l,c)))
  pairs=sorted(opt)[:8]; print('MORE',m,j,pairs[:3],flush=True)
