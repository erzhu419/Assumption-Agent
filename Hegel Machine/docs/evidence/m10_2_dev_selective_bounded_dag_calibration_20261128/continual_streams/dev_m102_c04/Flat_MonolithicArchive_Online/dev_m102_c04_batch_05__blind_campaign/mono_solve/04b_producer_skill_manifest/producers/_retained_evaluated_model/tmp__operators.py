import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; X=T['x']; Y=np.c_[T['y'],T['traj']]; time=s['eval_times']
def aff(*x):return np.column_stack((np.ones(len(x[0])),*x))
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def score(f):
 p=f@ls(f[:52],Y[:52]); e=abs(p-Y); return np.array([e[:52,0].mean(),e[52:,0].mean(),e[:52,1:].mean(),e[52:,1:].mean()])
d=s['panel_a']; u,v=d['u'],d['v']; mask=np.max(abs(v-u),axis=1)<3
def g(u,i):
 a,b=u[:,i],u[:,(i+1)%4];return aff(a,b,a*b,a**3)
wa=[ls(g(u,i)[mask],v[mask,i]) for i in range(4)]
def A(u):return np.column_stack([g(u,i)@wa[i] for i in range(4)])
def IA(x):return np.array([least_squares(lambda q:A(q[None])[0]-r,r,gtol=1e-13).x for r in x])
d=s['panel_b']; wb=ls(aff(d['u']),d['v']); B=lambda x:aff(x)@wb; IB=lambda x:(x-wb[0])@np.linalg.inv(wb[1:]);
pairs=list(itertools.combinations(range(4),2)); cross=lambda x: np.column_stack([x[:,i]*x[:,j] for i,j in pairs]);
d=s['panel_c']; u=d['u'];
funs={'ratodd':lambda a,k:a/(1+k*abs(a)), 'rateven':lambda a,k:a*a/(1+k*a*a), 'log':lambda a,k:np.log1p(k*a*a)/k,'tanh':lambda a,k:np.tanh(k*a),'sin':lambda a,k:np.sin(k*a),'cos':lambda a,k:np.cos(k*a),'softodd':lambda a,k:a/np.sqrt(1+k*a*a),'cubrat':lambda a,k:a/(1+k*a*a)}
base={'linear':lambda a:aff(a),'cross':lambda a:aff(cross(a)), 'sqcross':lambda a:aff(a*a,cross(a)), 'quad':lambda a:aff(a,a*a,cross(a)), 'cubeplus':lambda a:aff(a,a*a,a**3,cross(a))}
ops=[]
for key in ('v1','v2'):
 v=d[key]; results=[]
 def add(name,f,k=None):
  f0=f(u); w=ls(f0,v); err=np.mean(abs(f0@w-v)); results.append((err,name,k,lambda x,f=f,w=w:f(x)@w))
 for name,f in base.items():add(name,f)
 for name,g0 in funs.items():
  for mode in range(4):
   def feat(a,k,g0=g0,mode=mode):
    if mode==0:return aff(g0(a,k[0]))
    if mode==1:return aff(a,g0(a,k[0]))
    if mode==2:return aff(a+k[1]*g0(a,k[0]))
    return aff(cross(a),g0(a,k[0]))
   def err(k):
    f=feat(u,k);return (f@ls(f,v)-v).ravel()
   start=[.7,.5] if mode==2 else [.7]; lo=[.001,-4] if mode==2 else [.001];hi=[4,4] if mode==2 else [4]; k=least_squares(err,start,bounds=(lo,hi)).x;add(name+str(mode),lambda a,k=k,feat=feat:feat(a,k),k)
 results.sort(key=lambda z:z[0]);print(key,[(round(a,6),b,c) for a,b,c,_ in results[:12]]); ops.append(results[0][-1])
P,R=ops
if __name__=='__main__':
 z=IA(X); coords={'x':X,'z':z,'Bz':B(z),'IBz':IB(z),'Bx':B(X),'IBx':IB(X),'IAIBx':IA(IB(X)),'IABx':IA(B(X))}; out=[]
 for n,a in coords.items():
  p,r=P(a),R(a); d=(p*r).sum(1); c=np.einsum('ij,ik->ijk',p,r).reshape(len(a),-1)
  for key,f in [('pr',aff(p,r)),('prd',aff(p,r,d)),('prc',aff(p,r,c)),('prdc',aff(p,r,d,c[:,1])),('aprd',aff(a,p,r,d)),('pplus',aff(a,p)),('rplus',aff(a,r))]:out.append((score(f),n,key))
 for k in [1,3]:
  print('best',k, sorted(out,key=lambda q:q[0][k])[:10])
