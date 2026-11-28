exec(open('/tmp/read14.py').read().split('for f in (1.25')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
U=b['u']; V=b['v']; C=s['panel_c']; A=aff(U)
for key,p,q in [('b',b,'v'),('c1',C,'v1'),('c2',C,'v2')]:
 X=p['u']; Y=p[q];D=aff(X); print(key,X.shape,'baseline',np.std(D@fit(D,Y)-Y))
 for name,fn in [('sin',np.sin),('tanh',np.tanh),('rat',lambda x:x/(1+x*x)),('cos',np.cos)]:
  def e(theta):
   D1=aff(X+theta[0]*fn(theta[1]*X));return (D1@fit(D1,Y)-Y).ravel()
  vals=[least_squares(e,[w,f],bounds=([-20,.25],[20,5])) for w in [-.8,-.3,.3,1,3] for f in [1.25,2.5]]; o=min(vals,key=lambda o:o.cost);print('pre',name,o.x,np.std(o.fun))
  if key=='b' and name in ('sin','tanh','rat'):
   co=fit(aff(X+o.x[0]*fn(o.x[1]*X)),Y);globals()['z'+name]=aff(u+o.x[0]*fn(o.x[1]*u))@co
  def post(t):
   L=D@t[:-2].reshape(D.shape[1],-1);return (L+t[-2]*fn(t[-1]*L)-Y).ravel()
  o=least_squares(post,np.r_[fit(D,Y).ravel(),.22,1.25],max_nfev=200);print('post',name,o.x[-2:],np.std(o.fun))
  if key=='b' and name in ('sin','tanh','rat'):
   l=aff(u)@o.x[:-2].reshape(5,4);globals()['p'+name]=l+o.x[-2]*fn(o.x[-1]*l)
T=s['train']['traj']; yy=s['train']['y']; _,_,W=np.linalg.svd(T[:52],full_matrices=False); targets=np.c_[yy,T@W[:3].T];print('times',s['eval_times']);n=52

def poly(x,d):return np.column_stack([np.prod(x[:,q],axis=1) for q in cr(range(4),d)])
def check(name,z):
 Q=np.c_[aff(z),poly(z,2)]; j=fit(Q[:n],targets[:n]); print(name,'quad',np.round(j[:,0],3),np.round(j[:,1],3));
 for h in (0,1):
  t=targets[:,h]; vals=[]
  for base in (aff(z),Q):
   B=base[:n]; pinv=np.linalg.pinv(B); res=t[:n]-B@(pinv@t[:n]);rng=np.random.default_rng(55)
   for fn in (np.tanh,np.sin,np.cos,lambda p:p/(1+p*p),lambda p:p*p*p):
    def DD(q):return fn(aff(z)@q)
    def err(q):
     p=DD(q)[:n]; p-=B@(pinv@p);return res-p*((p@res)/max(p@p,1e-14))
    opts=[least_squares(err,v,max_nfev=110) for v in [np.r_[0,j[1:5,h]],np.r_[.1,np.eye(4)*2][0:5] if False else np.array([.1,1.,.6,1.,.6])]+list(rng.normal(0,1.25,(7,5)))]
    o=min(opts,key=lambda o:o.cost);D=np.c_[base,DD(o.x)]; co=fit(D[:n],t[:n]); e=abs(t-D@co);vals.append((e[:n].mean(),e[n:].mean(),fn.__name__,len(co),np.round(o.x,3),np.round(co,3)))
  print('profile',h,sorted(vals,key=lambda a:a[0])[:3])
for k in ('ztanh','psin','ptanh','prat'):check(k,globals()[k])
