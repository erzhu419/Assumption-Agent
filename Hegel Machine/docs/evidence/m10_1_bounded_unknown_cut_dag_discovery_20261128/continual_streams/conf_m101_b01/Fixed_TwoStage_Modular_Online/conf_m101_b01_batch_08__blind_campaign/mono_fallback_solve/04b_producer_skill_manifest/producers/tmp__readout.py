from compose import *
from sklearn.linear_model import Lasso, ElasticNet
from scipy.stats import pearsonr
u,v=(s['panel_b'][k] for k in ('u','v'))
def bf(p,x,kind):
 f,a,*b=p; h=x+a*np.sin(f*x)
 if kind=='square':h+=b[0]*x*x
 if kind=='cube':h+=b[0]*x**3
 if kind=='cos':h+=b[0]*(np.cos(f*x)-1)
 if kind=='second':h+=b[0]*np.sin(2*f*x)
 if kind=='prod':h+=b[0]*cycle(x)
 if kind=='sin2':h+=b[0]*np.sin(b[1]*x)
 if kind=='offset':h=x+a*np.sin(f*x+b[0])
 if kind=='tanh':h+=b[0]*np.tanh(b[1]*x)
 return aff(h)
def getb(kind):
 x0=[2.8,-.4]+([.01,1.2] if kind in ('sin2','tanh') else [.01] if kind!='base' else [])
 def fit(p):
  d=bf(p,u,kind);w=ls(d,v);return d@w-v,w
 p=least_squares(lambda p:fit(p)[0].ravel(),x0,max_nfev=250).x;r,w=fit(p);return np.mean(abs(r)),p,lambda x:bf(p,x,kind)@w
bs=[(k,*getb(k)) for k in ('base','square','cube','cos','second','prod','sin2','offset','tanh')]; print('B fits',[(k,e,p) for k,e,p,_ in bs]);
a=afit(s)['diagsecond'];z=inverse(a,s['train']['x']); op=operators(s);op['B']=min(bs,key=lambda b:b[1])[-1]; zs={'Z':z}
for k,f in op.items():zs[k]=f(z)
for p in ['B','C','D','BC','BD','CB','CD','DB','DC']:
 for k,f in op.items():
  if k not in p:zs[p+k]=f(zs[p]);
y=s['train']['y'];t=s['train']['traj'];print('ranges',[(k,round(np.max(abs(x)),2)) for k,x in zs.items()]);
def features(x,k):
 d={}
 def add(n,a):
  if a.ndim==1:a=a[:,None]
  for i,col in enumerate(a.T):d[f'{k}/{n}/{i}']=col
 r=R(x);ss=x+r;df=x-r;pr=x*r
 for n,a in [('x',x),('sq',x*x),('cycle',pr),('anti',x*np.roll(x,2,axis=1)[:,:]),('triple',pr*np.roll(x,2,axis=1)),('cube',x**3),('mixed',x*x*r),('mixed2',x*r*r),('tanh',np.tanh(2*x)),('sin',np.sin(3*x)),('cos',np.cos(2*x)),('rat',x/(1+2*x*x)),('pRat',pr/(1+ss*ss)),('pRat2',pr/(1+df*df)),('sRat',ss/(1+df*df)),('dRat',df/(1+ss*ss)),('hRat',x/(1+2*r*r)),('pTanh',np.tanh(2*pr)),('pSin',np.sin(3*pr))]:add(n,a)
 return d
F={k:features(x,k) for k,x in zs.items()};splits=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(3)];
for target in [y,t[:,-1]]:
 results=[]
 for group in [[k] for k in F]+[['Z','B'],['B','BC','BCD'],['Z','B','BC','BD','BCD'],list(F)]:
  for mode in ['linear','poly','rich']:
   names=[];cols=[]
   for k in group:
    for name,col in F[k].items():
     typ=name.split('/')[1];
     if mode=='linear' and typ!='x':continue
     if mode=='poly' and typ not in ['x','sq','cycle','anti','triple','cube','mixed','mixed2']:continue
     names.append(name);cols.append(col)
   d=np.array(cols).T
   for alpha in [.0005,.002,.005,.015,.035]:
    es=[]; co=None
    for n,te in enumerate(splits):
     tr=np.setdiff1d(np.arange(72),te);sc=np.maximum(d[tr].std(0),1e-9);m=ElasticNet(alpha=alpha,l1_ratio=.92,max_iter=4000,tol=.0003).fit(d[tr]/sc,target[tr]);es.append(np.mean(abs(m.predict(d[te]/sc)-target[te])));
     if n==0:co=[(names[i],round(m.coef_[i],3)) for i in np.argsort(abs(m.coef_))[-8:][::-1]]
    results.append((np.mean(es),es,group,mode,alpha,co))
 print('target',target.std());print(*sorted(results)[:8],sep='\n');print('gate',*sorted(results,key=lambda r:r[1][0])[:4],sep='\n')
