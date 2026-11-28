exec(open('/tmp/kerneltest.py').read().split('rows=[]')[0])
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
p=s['panel_c']; u=p['u'];
def design(u): return np.c_[np.ones(len(u)),u]
def panel(kind,v):
 def f(x,k):
  if kind=='atan':return design(np.arctan(k*x))
  return np.c_[design(x),x*x/(1+k*x*x)]
 def err(k):
  Q=np.linalg.qr(f(u,k))[0]; return np.mean(((v-Q@(Q.T@v))/(1-(Q*Q).sum(1)[:,None]))**2)
 k=minimize_scalar(err,bounds=(.02,2),method='bounded').x; W=np.linalg.lstsq(f(u,k),v,rcond=None)[0]; print(kind,k,err(k)**.5);return lambda x:f(x,k)@W
c1=panel('atan',p['v1']); c2=panel('rat',p['v2']); w=z['bf'];
base_y=_kernel(w,.18); base_t=_kernel(w,.18,.2)+.3*(z['forward']@z['forward'].T)**2
for name,a in z.items():
 if name in ('b','forward','inverse'): continue
 for typ,f in [('c1',c1),('c2',c2),('both',lambda x:np.c_[c1(x),c2(x)])]:
  q=f(a); q-=q.mean(0); k=q@q.T; d=cdist(q,q,'sqeuclidean'); d*=4/np.mean(d)
  for par in [.02,.1,.5,2.]:
   for r in [.001,.004,.01]:
    extra=par*(k+.2*k*k+np.exp(-.18*d)); py=evaluate(base_y+extra,r); pt=evaluate(base_t+extra,r)
    rows=globals().setdefault('rows',[]); rows.append(((name,typ,par,r),score(py),score(pt)))
for i in [1,2]:
 print('c sorted',i)
 for row in sorted(rows,key=lambda a:(a[i][1]+.25*a[i][2]+.2*a[i][3]) if i==1 else (a[i][5]+.25*a[i][6]+.2*a[i][7]))[:7]:print(row[0],row[i])
rows=[]
l=w@w.T; df=cdist(w,w,'sqeuclidean'); d0=cdist(z['forward'],z['forward'],'sqeuclidean');
for dname,d in [('bf',df),('mix',.6*df+.4*.77*d0),('fwd',.77*d0)]:
 for ww in [.07,.12,.18,.25,.35]:
  rr=np.sqrt(ww*d)
  for kname,k in [('rbf',np.exp(-rr*rr)),('rq5',(1+rr*rr/5)**-5),('rq1',(1+rr*rr)**-1),('m52',(1+rr+rr**2/3)*np.exp(-rr)),('m72',(1+rr+.4*rr**2+rr**3/15)*np.exp(-rr))]:
   for q in [0.,.15,.6]:
    for r in [.0003,.001,.003,.008]:
     P=evaluate(1+l+q*l*l+2*k,r);rows.append(((dname,ww,kname,q,r),score(P)))
for i in [0,4]:
 print('stationary',i)
 for row in sorted(rows,key=lambda a:a[1][i+1]+.25*a[1][i+2]+.2*a[1][i+3])[:10]:print(*row)
