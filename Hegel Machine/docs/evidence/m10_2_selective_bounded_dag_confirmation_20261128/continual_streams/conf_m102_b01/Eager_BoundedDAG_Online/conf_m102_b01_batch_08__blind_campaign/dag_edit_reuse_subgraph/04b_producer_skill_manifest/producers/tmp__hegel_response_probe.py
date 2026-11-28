exec(open('/tmp/hegel_panel_structure_probe.py').read().split('for name,x,v in')[0])
from scipy.optimize import least_squares
X=b['u']; V=b['v']; cs=c['u']; n=len(s['train']['y']); Y=s['train']['y']; T=s['train']['traj']; t=np.arange(n); splits=[(t[:52],t[52:])]+[(np.delete(t,i),i) for i in np.array_split(np.random.default_rng(57).permutation(t),4)]
def fb(x,g): return np.c_[pol(x,1), x**3/(1+g*x*x)]
for typ in ['rational','tanh']:
 def mk(x,g):return np.c_[pol(x,1), x**3/(1+g*x*x) if typ=='rational' else np.tanh(x*g)]
 def fun(g):
  d=mk(X,g);return (d@ls(d,V)-V).ravel()
 opt=least_squares(fun,[1.]*4,bounds=(.01,4)); print('B multi',typ,opt.x,np.sqrt(np.mean(opt.fun**2)),np.round(ls(mk(X,opt.x),V),4));
 if typ=='rational':W=mk(u,opt.x)@ls(mk(X,opt.x),V)
def c1(x):return np.c_[np.ones(len(x)),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]]
C1=c1(u)@ls(c1(cs),c['v1']);C2=pol(u,2)@ls(pol(cs,2),c['v2']); print('trajectory svd',np.linalg.svd(T,compute_uv=False),'time',s['eval_times']);
def ev(F,Y):
 vals=[]
 for i,j in splits:
  f=F/np.where(F[i].std(0)>1e-10,F[i].std(0),1);g=f[i].T@f[i];h=f[i].T@Y[i];vals.append([np.mean(abs(f[j]@np.linalg.solve(g+np.eye(len(g))*l,h)-Y[j])) for l in [.001,.05,.3,2,5]])
 vals=np.array(vals);k=vals[1:].mean(0).argmin();return vals[1:,k].mean(),vals[:,k].round(4),k
for y in [Y,T]:
 ans=[]
 for name,Z in [('B',W),('C1',C1),('C2',C2),('B+C1',np.c_[W,C1]),('B+C2',np.c_[W,C2]),('C1+C2',np.c_[C1,C2]),('u+B',np.c_[u,W]),('all',np.c_[W,C1,C2])]:
  for trans,f in [('id',lambda x:x),('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('rat',lambda x:x/(1+x*x))]:
   for g in ([1] if trans=='id' else [.5,1,2,4,6]):
    H=f(g*Z)
    for d in (1,2,3) if Z.shape[1]==4 else (1,2):
     for add in ['','other','base','full']:
      F=pol(H,d)
      if add=='other':F=np.c_[F,C1,C2]
      if add=='base':F=np.c_[F,Z]
      if add=='full':F=np.c_[F,W,C1,C2,u]
      e=ev(F,y);ans.append((e[0],e[1],name,trans,g,d,add,e[2]))
 print('best response',y.shape);print(*sorted(ans,key=lambda q:q[0])[:14],sep='\n')
np.savez('/tmp/hegel_calibrated.npz',u=u,W=W,C1=C1,C2=C2)
