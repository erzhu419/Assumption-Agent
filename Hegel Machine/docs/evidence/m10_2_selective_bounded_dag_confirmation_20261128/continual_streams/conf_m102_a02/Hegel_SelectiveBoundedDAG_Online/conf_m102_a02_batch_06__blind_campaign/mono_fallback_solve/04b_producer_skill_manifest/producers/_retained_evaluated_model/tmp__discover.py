exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
ls=lambda f,v:lstsq(f,v)[0]
np.set_printoptions(precision=4,suppress=True)
W={'q':q}; M={}
for n,p,key in [('b','b','v'),('c','c','v1'),('d','c','v2')]:
 P=s['panel_'+p]; U,V=P['u'],P[key]
 fam={'power':(lambda a,k:a*np.exp(k[2]*np.logaddexp(0,k[1]*np.log(np.maximum(abs(a)*k[0],1e-50)))),[2.5,12,.036],[.1,1,.001],[10,40,2]),'powered':(lambda a,k:a*(1+k[0]*a*a)**k[1],[1,-.5],[.02,-4],[25,4]),'asinh':(lambda a,k:np.arcsinh(k[0]*a),[2],[.02],[15]),'softabs':(lambda a,k:np.sqrt(1+k[0]*a*a),[1],[.02],[15]),'tanh':(lambda a,k:np.tanh(k[0]*a),[1.25],[.02],[15])}
 opts=[]
 for nm,(h,k0,lo,hi) in fam.items():
  def f(a,k,h=h):return np.c_[np.ones(len(a)),a,h(a,k)]
  def r(k):return (f(U,k)@ls(f(U,k),V)-V).ravel()
  k=least_squares(r,k0,bounds=(lo,hi),max_nfev=350).x; F=f(U,k); B=ls(F,V); e=F@B-V; opts.append((np.std(e),nm,k,B,lambda a,k=k,B=B,f=f:f(a,k)@B))
 opts.sort(key=lambda t:t[0]);print(n,[o[:3] for o in opts],opts[0][3],flush=True); M[n]=opts[0][-1];W[n]=M[n](q)
for i,mi in M.items():
 for j,mj in M.items():
  if i!=j:
   W[j+i]=mi(W[j]); W[i+'q'+j]=mi(q+W[j]);W[j+i+j]=mj(W[j+i]);
print('times',t,'yt',np.corrcoef(np.c_[y,z].T)[0]); print('summary',tr['summary'],'samples',np.c_[q,y,z][:7]);
P=lambda a,d:PolynomialFeatures(d).fit_transform(a)
Y=np.c_[y,z]; out=[]
for n,a in W.items():
 for k,b in [('none',a[:,:0]),*[(k,W[k]) for k in ['q','b','c','d'] if k!=n]]:
  for d in [1,2,3]:
   for typ in ['poly','add','cross']:
    if typ!='poly' and (d==1 or k=='none'):continue
    f=P(np.c_[a,b],d) if typ=='poly' else np.c_[P(a,d),P(b,d)[:,1:],((a[:,:,None]*b[:,None,:]).reshape(len(a),-1) if typ=='cross' else a[:,:0])];
    for alp in [1e-8,1e-5,1e-3,.02,.2]:
     g=f/np.maximum(f[:52].std(0),.1); pred=Ridge(alpha=alp).fit(g[:52],Y[:52]).predict(g); r=abs(Y-pred); out.append((r[52:,0].mean(),r[52:,1:].mean(),r[:52,0].mean(),r[:52,1:].mean(),n,k,d,typ,alp))
for i in [0,1]:
 print('best',i);print(*sorted(out,key=lambda a:a[i])[:15],sep='\n')
np.savez('/tmp/discover.npz',**W)
