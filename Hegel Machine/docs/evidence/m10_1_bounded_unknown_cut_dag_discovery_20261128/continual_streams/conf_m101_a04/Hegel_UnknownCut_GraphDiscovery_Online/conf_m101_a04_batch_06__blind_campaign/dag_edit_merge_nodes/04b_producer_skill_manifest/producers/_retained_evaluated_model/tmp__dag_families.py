exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
from scipy.linalg import qr
from itertools import combinations_with_replacement as cwr
cb=lambda q:np.c_[np.ones(len(q)),np.cos(2.4*q)]
c=(cb(x)@np.linalg.lstsq(cb(u),p['v2'],rcond=None)[0])[:n]; x=x[:n];b=b[:n]; Y=np.c_[y,tr]; k=52
fit=lambda a,y:np.linalg.lstsq(a,y,rcond=1e-9)[0]
def ev(A,verbose=True):
 A=np.c_[np.ones(n),A]; res=Y-A@fit(A[:k],Y[:k]); E=np.abs(res); return np.round([E[:k,0].mean(),E[k:,0].mean(),E[:k,1:].mean(),E[k:,1:].mean()],5), np.round(np.linalg.svd(res[:k],compute_uv=False)[:4],4)
print('keys B',list(s['panel_b']), 'keys study',list(s));
B=np.c_[b,c]; C=np.cos(2.4*x); Z=np.c_[x,b,c]; a=np.c_[np.ones(n),B]
D={}
for g in [.5,1,1.5,2,2.4,3,4]:
 for nm,q in [('b',b),('c',c),('b+c',b+c),('b-c',b-c),('bc',b*c),('bxc',b[:,:,None]*c[:,None,:]),('cp',C),('cp*2',C**2)]:
  q=q.reshape(n,-1)
  for f in [np.sin,np.cos,np.tanh,lambda z: z/(1+z*z)]: D[f'{f.__name__} {g} {nm}']=f(g*q)
 for nm,aa,bb in [('cb',c,b),('bc',b,c),('bb',b,b),('cc',c,c),('xc',x,c),('cx',c,x),('xb',x,b),('bx',b,x)]:
  D[f'rat {g} {nm}']=aa/(1+g*bb**2)
  D[f'ratall {g} {nm}']=(aa[:,:,None]/(1+g*bb[:,None,:]**2)).reshape(n,-1)
  D[f'gate {g} {nm}']=aa*np.tanh(g*bb)
  D[f'gateall {g} {nm}']=(aa[:,:,None]*np.tanh(g*bb[:,None,:])).reshape(n,-1)
for nm,q in [('b',b),('c',c),('bc',B),('x',x),('Z',Z)]:
 for deg in [2,3,4]:
  ix=list(cwr(range(q.shape[1]),deg));D[f'poly{deg} {nm}']=np.prod(q[:,ix],axis=2)
D['elem b*c']=b*c;D['elem x*c']=x*c;D['elem x*b']=x*b
for name in sorted(D,key=lambda nm:sum(ev(np.c_[B,D[nm]])[0][[1,3]]))[:24]:print(name,ev(np.c_[B,D[name]]))
# Fit multiple shared ridge functions with variable projection; a product of
# affine projections is included as a separate family.
Q=qr(a[:k],mode='economic')[0]; Y0=Y[:k]-Q@(Q.T@Y[:k]); rng=np.random.default_rng(21)
for nm,P in [('bc',B),('b and c separately',B),('x and bc',Z)]:
 W=P/P[:k].std(axis=0); W0=W[:k]; d=W.shape[1]
 for funs in [(np.tanh,np.tanh,np.tanh),(np.sin,np.sin,np.sin),(np.tanh,np.sin,np.cos)]:
  def F(z,W):
   H=W@z[:,:d].T+z[:,d];return np.column_stack([f(h) for f,h in zip(funs,H.T)])
  def obj(z):
   H=F(z.reshape(3,d+1),W0); H-=Q@(Q.T@H);return (Y0-H@fit(H,Y0)).ravel()
  best=None
  for i in range(12):
   ini=rng.normal(0,.4,(3,d+1));r=least_squares(obj,ini.ravel(),max_nfev=85); sc=np.linalg.norm(r.fun)
   if best is None or sc<best[0]:best=sc,r.x.reshape(3,d+1)
  print('multiridge',nm,[f.__name__ for f in funs],round(best[0],4),ev(np.c_[B,F(best[1],W)])[0],flush=True)
