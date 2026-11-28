exec(open('/tmp/panel_scan.py').read().split('for name,q in Q.items():')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
splits=[np.arange(52,72),np.arange(0,20),np.arange(26,46),np.random.default_rng(726).permutation(len(x))[:20]]
for ii,va in enumerate(splits):
 ti=np.setdiff1d(np.arange(len(x)),va); yy=z[ti]; _,ss,vt=np.linalg.svd(yy,full_matrices=False); yy=yy@vt[:2].T
 def loss(pred):return np.mean(abs(pred@vt[:2]-z[va]))
 print('FOLD',ii,flush=True)
 for name in ['raw','Av','Bw','Bwv']:
  q=Q[name]; pred=np.column_stack([gp(q[ti],q[va],y,True) for y in yy.T]); print('GP',name,loss(pred),flush=True)
 q=(x-x[ti].mean(0))/x[ti].std(0)
 for f in ('sin','tanh','exp','rat','atan'):
  for d in (1,2):
   for m in (1,2,3):
    fn={'sin':np.sin,'tanh':np.tanh,'exp':lambda v:np.exp(-v*v), 'rat':lambda v:v/(1+v*v), 'atan':np.arctan}[f]
    base=np.c_[np.ones(len(x)),q];
    if d==2:base=np.c_[base,*[q[:,j]*q[:,k] for j,k in combinations_with_replacement(range(4),2)]]
    aq=np.c_[np.ones(len(q)),q]; target=yy[:,0]; penalty=0.012
    def calc(p):
     dd=np.c_[base,fn(aq@p.reshape(m,5).T)]; X=dd[ti]; R=np.eye(X.shape[1])*penalty; c=np.linalg.solve(X.T@X+R,X.T@target); return dd,c
    def obj(p):
     dd,c=calc(p); return np.r_[dd[ti]@c-target,np.sqrt(penalty)*c,.012*p]
    rng=np.random.default_rng(23); opts=[]
    for j in range(5):
     fitp=least_squares(obj,rng.normal(0,.7,(m,5)).ravel(),max_nfev=110,ftol=1e-6); opts.append((fitp.cost,fitp.x))
    cost,p=min(opts,key=lambda o:o[0]); dd,c=calc(p); p2=gp(q[ti],q[va],yy[:,1],True); print(f,d,m,round(cost,5),round(loss(np.c_[dd[va]@c,p2]),5),flush=True)
