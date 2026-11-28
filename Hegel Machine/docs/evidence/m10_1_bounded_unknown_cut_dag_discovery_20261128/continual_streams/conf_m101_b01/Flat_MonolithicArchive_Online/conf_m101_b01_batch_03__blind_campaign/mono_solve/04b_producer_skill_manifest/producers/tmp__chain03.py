from discover03 import *
from scipy.special import hyp2f1,erf

def a_design(q,j):
 a=q[:,j]; b=q[:,(j+1)%4]; return np.c_[np.ones(len(q)),a,b,a*b,a**3]
idx=e<1
AA=np.column_stack([L(a_design(u[idx],j),v[idx,j],rcond=None)[0] for j in range(4)])
def calibrate(q):
 cur=(q-A[0])@np.linalg.inv(A[1:]); j=np.arange(4); k=(j+1)%4
 for _ in range(20):
  J=np.zeros((len(q),4,4)); J[:,j,j]=AA[1]+AA[3]*cur[:,k]+3*AA[4]*cur**2; J[:,j,k]=AA[2]+AA[3]*cur
  err=np.column_stack([a_design(cur,i)@AA[:,i] for i in j])-q
  step=np.linalg.solve(J,err[:,:,None])[:,:,0];cur-=step/np.maximum(1,np.max(abs(step),axis=1,keepdims=True)/.3)
 return cur,np.max(abs(err),axis=1)
uc=pc['u']; c1=pc['v1']; c2=pc['v2']
phis={'atan':lambda q,p:np.arctan(q*np.exp(p[0])), 'asinh':lambda q,p:np.arcsinh(q*np.exp(p[0])), 'tanh':lambda q,p:np.tanh(q*np.exp(p[0])), 'erf':lambda q,p:erf(q*np.exp(p[0])), 'rational':lambda q,p:q/(1+np.exp(p[0])*q*q)**np.exp(p[1]), 'integral':lambda q,p:q*hyp2f1(.5,np.exp(p[1]),1.5,-np.exp(p[0])*q*q), 'softsign':lambda q,p:q/(1+np.exp(p[0])*abs(q))}
fits={}
for n,phi in phis.items():
 def make(p,q):return D(phi(q,p))
 def fun(p):
  m=make(p,uc);return (m@L(m,c1,rcond=None)[0]-c1).ravel()
 f=least_squares(fun,[0.,0.] if n in ('rational','integral') else [0.],bounds=(-4,4),max_nfev=100)
 fits[n]=(f.x,L(make(f.x,uc),c1,rcond=None)[0]); print('C1',n,f.x,np.std(f.fun))

def p4(q):return np.c_[D(q),q*q,q**3,q**4]
C4=L(p4(uc),c2,rcond=None)[0]; print('A coefs',AA,'A inlier rms',np.sqrt(np.mean((np.column_stack([a_design(u[idx],i)@AA[:,i] for i in range(4)])-v[idx])**2))); print('C2 powers',np.round(C4,4));
def C1(q,n):
 p,w=fits[n];return D(phis[n](q,p))@w

def c2fit(name,phi,p0):
 def fun(p):
  m=D(phi(uc,p));return (m@L(m,c2,rcond=None)[0]-c2).ravel()
 f=least_squares(fun,p0,bounds=(-5,5),max_nfev=100)
 print('C2 sep',name,f.x,np.std(f.fun));return lambda q:D(phi(q,f.x))@L(D(phi(uc,f.x)),c2,rcond=None)[0]
C2s={
 'poly':c2fit('poly',lambda q,p:q+p[0]*q*q+p[1]*q**3+p[2]*q**4,[.45,0,0]),
 'tanh':c2fit('tanh',lambda q,p:np.tanh(np.exp(p[0])*(q+p[1])),[0,-.8]),
 'erf':c2fit('erf',lambda q,p:erf(np.exp(p[0])*(q+p[1])),[0,-.8]),
 'sqrat':c2fit('sqrat',lambda q,p:(q+p[1])/(1+np.exp(p[0])*(q+p[1])**2)**np.exp(p[2]),[0,-.8,-.6]),
 'squared':c2fit('squared',lambda q,p:q+p[0]*q**2+p[1]*(q+p[0]*q**2)**2,[.45,.02]),
 'ratquad':c2fit('ratquad',lambda q,p:(q+p[0]*q*q)/(1+np.exp(p[1])*q*q),[.45,-4]),
}
zz,ee=calibrate(x); print('inverse errors',np.quantile(ee,[0,.9,1]),'bounds',zz.min(0),zz.max(0)); bb=D(zz)@B
H=timefit(T[:52],'2exp')[1]; V=np.c_[y,T@np.linalg.pinv(H.T)]
coords={'a':zz,'ab':bb}
for tag,q in list(coords.items()):
 for n in ('rational','integral','asinh'):coords[tag+'1'+n]=C1(q,n)
 coords[tag+'2']=p4(q)@C4;coords[tag+'2sep']=C2s['poly'](q)
 if tag=='ab':
  for n,fn in C2s.items():coords[tag+'2'+n]=fn(q)
for key in [k for k in coords if k.startswith('ab1') or k=='ab2']:
 q=coords[key];coords[key+'2']=p4(q)@C4;coords[key+'1']=C1(q,'rational')
res=[]
for k,q in coords.items():
 for typ,G in [('lin',D(q)),('quad',F(q,2)),('cub',F(q,3)),('tanh',F(np.tanh(q),2)),('sin',F(np.sin(q),2)),('sep3',np.c_[D(q),q**2,q**3]),('pair',np.c_[D(q),q[:,:2]*q[:,2:]]),('quart',p4(q))]:
  p=G@L(G[:52],V[:52],rcond=None)[0]; res.append((np.mean(abs(p[52:,0]-y[52:])),np.mean(abs(p[52:,1:]@H.T-T[52:])), k, typ,np.mean(abs(p[:52]-V[:52]),0)))
print('best y',*sorted(res,key=lambda a:a[0])[:12],sep='\n');print('best traj',*sorted(res,key=lambda a:a[1])[:12],sep='\n')
