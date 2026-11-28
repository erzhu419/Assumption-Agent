exec(open('/tmp/probe6.py').read().split('for xx,n')[0]); from itertools import combinations
from sklearn.linear_model import orthogonal_mp
ops={'rational':(lambda x,a,b:x/(1+a*x*x)**b,[1,.5]),'soft':(lambda x,a,b:x/(1+a*abs(x))**b,[1,1]),'softp':(lambda x,a,b:x/(1+a*abs(x)**b),[1,1.5]),'atan':(lambda x,a,b:np.arctan(a*x)+b*x,[2,0]),'tanh':(lambda x,a,b:np.tanh(a*x)+b*x,[1.6,0]),'sin':(lambda x,a,b:np.sin(a*x)+b*x,[1.2,0]),'erf':(lambda x,a,b:__import__('scipy').special.erf(a*x)+b*x,[1.5,0])}
j=np.arange(len(u))%5!=0; warps={}
for name,(op,p0) in ops.items():
 def f(p,xx):return h(op(xx,p[0],p[1]))
 lo=[.01,.01] if name in ['rational','soft','softp'] else [.01,-4]; hi=[10,8]; res=least_squares(lambda p:(f(p,u[j])@fit(f(p,u[j]),v[j])-v[j]).ravel(),p0,bounds=(lo,hi)); print(name,res.x,rms(res.fun),rms(f(res.x,u[~j])@fit(f(res.x,u[j]),v[j])-v[~j])); warps[name]=op(z,*res.x)
warps.update(raw=z,sin12=np.sin(1.2*z),tanh12=np.tanh(1.2*z),atan21=np.arctan(2.1*z),rational125=z/(1+1.25*z*z)**.5,soft125=z/(1+1.25*abs(z)))
def dicts(a,N):
 fs=[(np.ones(len(a)), '1')]
 def add(x,n):fs.append((x,N+':'+n))
 for i,p in enumerate(a.T):
  for k in range(1,5):add(p**k,f'{i}^{k}')
  for nm,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('soft',lambda p:p/(1+abs(p))),('r1',lambda p:p/(1+p*p)),('e',lambda p:np.exp(-p*p))]:
   for sc in [.7,1,1.25,1.5,2,3]:add(f(sc*p),f'{nm}({sc}*{i})')
 for i,j in combinations(range(4),2):
  p,q=a[:,i],a[:,j]
  for k,l in [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]:add(p**k*q**l,f'{i}^{k}*{j}^{l}')
  for sc in [.25,.5,1,1.25,2]:
   add(p*q/(1+sc*(p*p+q*q)),f'pair{i}{j}r{sc}')
   for sg in [-1,1]:
    r=p+sg*q
    for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda r:r/(1+r*r)),('e',lambda r:np.exp(-r*r))]:add(fn(sc*r),f'{nm}{sc}({i},{sg*j})')
 for tup in combinations(range(4),3):add(np.prod(a[:,tup],1),str(tup))
 return fs
sp=[(np.arange(52),np.arange(52,len(z)))]+[(np.where(np.arange(len(z))%4!=j)[0],np.where(np.arange(len(z))%4==j)[0]) for j in range(4)]
for k in range(3):
 best=[]
 for N,a in warps.items():
  D=dicts(a,N); F=np.array([f for f,n in D[1:]]).T
  for n in [3,5,8,12,16,21]:
   es=[]; terms=[]
   for i,j in sp:
    m=F[i].mean(0); sd=np.maximum(np.linalg.norm(F[i]-m,axis=0),1e-9); G=(F-m)/sd; yy=Y[i,k]-Y[i,k].mean(); c=orthogonal_mp(G[i],yy,n_nonzero_coefs=n); p=G[j]@c+Y[i,k].mean(); es.append(np.mean(abs(p-Y[j,k])));terms.append(np.flatnonzero(c))
   best.append((max(es),np.mean(es),N,n,np.round(es,4),[D[l+1][1] for l in terms[0]]))
 print('BEST',k,*sorted(best,key=lambda q:q[0])[:5],sep='\n')
