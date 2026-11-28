from explore import *
np.set_printoptions(precision=5,suppress=True)
u=s['panel_c']['u']; v=s['panel_c']; bp=lin(one(pb['u']),pb['v']); print('B',bp)
def maker(y,g,init):
 def fun(q):
  f=g(u,q[0]);return (f@lin(f,y)-y).ravel()
 q=least_squares(fun,[init],bounds=(.04,3),gtol=1e-13).x[0]; F=g(u,q); co=lin(F,y); print('panel q, co, rmse',q,co,np.std(fun([q])));return lambda x:g(x,q)@co
c=maker(v['v1'],lambda x,q:one(np.c_[x,x/(1+q*abs(x))]),.85)
e=maker(v['v2'],lambda x,q:one(np.c_[x,x*x,x*x/(1+q*x*x)]),.3)
B=lambda x:one(x)@bp
from scipy.linalg import svd
from itertools import combinations
G,H=c(Z),e(Z); t0=np.array([.42,1.82]); F=basis(t0); U=lin(F,T.T).T; D=np.c_[Y,U]; train=np.arange(52); val=np.arange(52,72)
def show(p):
 f=one(p); co=lin(f[train],D[train]); r=D-f@co; return np.round(np.r_[np.std(r[train],0),np.mean(abs(r[val,0])),np.mean(abs(r[val,1:]@F.T))],5),co,r
P=np.einsum('ni,nj->nij',G,H).reshape(N,16); Q=np.c_[G,H]; base,co,r=show(np.c_[Q,P]);print('bilinear',base,'co',co[:9]);
for j in range(3):
 m=co[9:,j].reshape(4,4); print(j,m,svd(m,compute_uv=False))
print('resids vs inputs',np.c_[Z,G,H,r][np.argsort(abs(r[train,0]))[-12:]])
print('summary',d['summary']); print('corr',np.corrcoef(np.c_[D,Q].T)[:3]);
registry=[]
def rec(tag,f):
 z,_,_=show(f);registry.append((tag,z))
for nm,k,l in [('GH',G,H),('HG',H,G),('ZG',Z,G),('ZH',Z,H),('GDS',G,G+H),('GDD',G,G-H),('HDS',H,G+H),('HDD',H,G-H),('DOT',P,P)]:
 if nm=='DOT':continue
 for p in [.1,.22,.4,.7,1.25,2.]:
  for den,nm2 in [(l*l,'s'),(abs(l),'a'),(np.sum(l*l,axis=1,keepdims=True),'norm'),(np.sum(k*l,axis=1,keepdims=True)**2,'dot')]:
   r=k/(1+p*den); rec((nm,nm2,p),np.c_[Q,r,P]); rec((nm,nm2,p,'small'),np.c_[Q,r,k*l])
  for name,w in [('tanh',np.tanh(p*(k+l))),('sin',np.sin(p*(k+l))),('soft',(k+l)/(1+p*abs(k+l))),('rat',(k+l)/(1+p*(k+l)**2)),('cube',(k+l)**3)]:rec((nm,name,p),np.c_[Q,w,P])
for i in [3,4]:
 print('best',i);print(*sorted(registry,key=lambda r:r[1][i])[:12],sep='\n')
