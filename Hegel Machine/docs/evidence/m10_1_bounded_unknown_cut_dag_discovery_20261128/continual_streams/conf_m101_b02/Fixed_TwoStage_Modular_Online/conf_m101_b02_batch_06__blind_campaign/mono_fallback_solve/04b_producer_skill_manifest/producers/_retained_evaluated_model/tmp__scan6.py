exec(open('/tmp/probe.py').read().split('print(\'A RMSE')[0])
from scipy.optimize import root, least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.special import erf
ai=np.array([root(lambda z:A(z[None])[0]-v,v).x for v in x]);
p=s['panel_b']; u,v=p['u'],p['v']; pc=s['panel_c'];uc=pc['u']; vc=pc['v2'];
for name,f in [('tanh',np.tanh),('erf',erf),('sin',np.sin)]:
 for extra,ex in [('none',lambda z:z[:,:0]),('sq',lambda z:z*z),('cross',q),('fourth',lambda z:z**4)]:
  def g(z,k):return des(z,f(k*z),ex(z))
  def r(k):
   z=g(u,k);return (z@ls(z,v)-v).ravel()
  k=minimize_scalar(lambda k:np.linalg.norm(r(k)),bounds=(.12,4),method='bounded').x;Z=g(u,k); P=Z@np.linalg.pinv(Z); print(name,extra,round(k,5),round(np.std(r(k)),7),round(np.std(r(k).reshape(v.shape)/(1-P.diagonal()[:,None])),7),flush=True)
  if name=='erf' and extra=='none':
   b=ls(Z,v); B=lambda z,k=k,b=b:des(z,erf(k*z))@b
k=minimize_scalar(lambda k:np.linalg.norm((lambda z:z@ls(z,vc)-vc)(des(uc,uc**2/np.sqrt(1+k*uc**2)))),bounds=(.01,2),method='bounded').x; D=fit(lambda u:des(u,u**2/np.sqrt(1+k*u*u)),'panel_c','v2')
# Screen coordinate chains separately on independent subsets of the first 52.
regs=np.array([1e-7,1e-5,.001,.01,.08]); res=[]; splits=[(np.arange(40),np.arange(40,52)),(np.setdiff1d(np.arange(52),np.arange(0,52,4)),np.arange(0,52,4)),(np.arange(52),np.arange(52,72))];
def scan(name,z,rich=True):
 z=(z-z[:40].mean(0))/np.maximum(z[:40].std(0),.03); d=z.shape[1]; L=z@z.T/d; Q=cdist(z,z,'sqeuclidean')/d; add=lambda f:sum(f(z[:,i,None]-z[None,:,i])/d for i in range(d)); ks=[('linear',L),('quad',(1+L)**2),('cubic',(1+L)**3),('mix4',L+.3*L**2+.08*L**3+.03*L**4),('add', L+add(lambda x:np.exp(-x*x/2))+.25*L**2)];
 if rich:
  for gamma in (.08,.25,.7):
   E=np.exp(-gamma*Q); delta=np.sqrt(6*gamma*Q);ks.extend([(f'rbf{gamma}',L+.3*L**2+E),(f'm32{gamma}',L+.3*L**2+(1+delta)*np.exp(-delta))])
 for kn,K in ks:
  allerr=[]
  for tr,te in splits:
   block=K[np.ix_(tr,tr)]+1; cross=K[np.ix_(te,tr)]+1; w,U=eigh(block); T=U.T@Y[tr]; F=cross@U
   e=np.array([np.mean(abs((F/(w+l))@T-Y[te]),axis=0) for l in regs]); allerr.append(np.c_[e[:,0],e[:,1:].mean(1)])
  o=(allerr[0]+allerr[1])/2
  for j in range(len(regs)):res.append((name,kn,regs[j],o[j],allerr[2][j]))
for n,z in [('x',x),('a',ai)]:
 bank={n:z}
 for sn,f in [('B',B),('C',C),('D',D)]:
  bank[n+sn]=f(z)
  for sn2,f2 in [('B',B),('C',C),('D',D)]:
   if sn!=sn2:bank[n+sn+sn2]=f2(f(z))
 for a,b in bank.items():
  scan(a,b); 
  if a!=n:scan(n+'+'+a,np.c_[z,.65*b],False)
 for sn,base in [(n,z),(n+'B',B(z)),(n+'C',C(z)),(n+'D',D(z))]:scan(sn+'+CD',np.c_[base,C(base),D(base)])
scan('x+a',np.c_[x,ai]);scan('x+a+ops',np.c_[x,ai,B(ai),C(ai),D(ai)])
for i in (0,1):
 print('inner ranked',i);print(*sorted(res,key=lambda r:r[3][i])[:13],sep='\n');print('outer ranked',i);print(*sorted(res,key=lambda r:r[4][i])[:8],sep='\n')
