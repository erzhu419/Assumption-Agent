from explore import *
from scipy.optimize import least_squares
from scipy.linalg import lstsq
p=PolynomialFeatures(2).fit_transform(x)
folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(26).permutation(72),4)]
forms={'tanh':np.tanh,'sin':np.sin,'rat':lambda a:a/(1+a*a),'exp':lambda a:np.exp(-a*a),'soft':lambda a:np.logaddexp(0,a)}
def run(tr,te,name,kind,nstarts=14):
 rng=np.random.default_rng(4890); h=p[tr]; yy=y[tr]; U=np.linalg.qr(h)[0]; yy0=yy-U@(U.T@yy); gx=x[tr]; fun=forms[name]; best=None
 def features(a,q,pp):
  r=fun(q@a[:4]+a[4]);
  if kind=='one':return r[:,None]
  if kind=='pair':return np.c_[r,fun(q@a[5:9]+a[9])]
  if kind=='product':return r[:,None]*np.c_[np.ones(len(q)),q]
  return r[:,None]*pp
 def fit(a):
  f=features(a,gx,h); f-=U@(U.T@f); b=lstsq(f,yy0,cond=1e-7)[0]; return yy0-f@b
 for j in range(nstarts):
  a=rng.normal(0,1.1,10 if kind=='pair' else 5)
  if j<6:
   a[:4]=coords['AiB'][1:5].std() if False else (B[1:,j%4] if j<4 else np.linalg.lstsq(np.c_[np.ones(len(tr)),gx],yy,rcond=None)[0][1:]); a[:4]*= 1.5/np.linalg.norm(a[:4]); a[4]=0 if j<5 else .4
  r=least_squares(fit,a,bounds=(-6,6),max_nfev=105,ftol=1e-7)
  sc=np.mean(r.fun**2)
  if best is None or sc<best[0]: best=sc,r.x
 sc,a=best; f=np.c_[h,features(a,gx,h)]; b=lstsq(f,yy,cond=1e-7)[0]; pred=np.c_[p[te],features(a,x[te],p[te])]@b
 return np.sqrt(sc),np.abs(pred-y[te]).mean(),a,b
if __name__=='__main__':
 for n in forms:
  for k in ('one','pair','product','quotient'):
   if k=='quotient' and n not in ('exp','rat'): continue
   results=[run(tr,te,n,k) for tr,te in folds[:2]]
   a,b=results[0][2:]; print(n,k,[(round(i,5),round(j,5)) for i,j,*r in results], 'a',a.round(3),'b',b.round(3),flush=True)
