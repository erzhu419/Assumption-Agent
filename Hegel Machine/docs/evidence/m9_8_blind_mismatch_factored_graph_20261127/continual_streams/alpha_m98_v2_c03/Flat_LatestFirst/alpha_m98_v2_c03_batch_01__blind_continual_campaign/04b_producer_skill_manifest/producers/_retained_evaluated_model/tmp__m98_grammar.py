exec(open('/tmp/m98_amplitude.py').read().split("print('rank-one")[0])
from scipy.optimize import least_squares
lab=ch['lab_id']; raw=ch['coupon_features_raw']; r=drift@b[2:]; lin=r-b[2]; w=b[3:]; N=len(r)
print('refs ranges and missing',[(int(j),np.round(ch['ref_coupons'][lab==j].mean(),4),np.round(ch['ref_coupons'][lab==j].std(),4)) for j in np.unique(lab)],np.isnan(raw).sum(0)); print('joint-standalone',np.round(x-old._features(ch,ch)[:N],3)); print('fit y =',np.round(target,4))
def evaluate(A,y=target):
 c=np.linalg.lstsq(A,y,rcond=1e-9)[0]; e=A@c-y; es=np.empty(N); labs=np.empty(N)
 for i in range(N):
  k=np.arange(N)!=i; es[i]=A[i]@np.linalg.lstsq(A[k],y[k],rcond=1e-9)[0]-y[i]
 for g in np.unique(lab):
  k=lab!=g; labs[~k]=A[~k]@np.linalg.lstsq(A[k],y[k],rcond=1e-9)[0]-y[~k]
 return np.sqrt(np.mean(e*e)),np.sqrt(np.mean(es*es)),np.max(abs(es)),np.sqrt(np.mean(labs*labs)),c

def pr(name,A,y=target):
 a,l,m,ll,c=evaluate(A,y); print(name,np.round([a,l,m,ll],5),np.round(c,5))
for tag,xx in [('joint',x),('diag',old._features(ch,ch)[:N]),('train+diag',old._features(tr,ch)[-N:])]:
 rr=b[2]+xx@w; dd=np.c_[np.ones(N),xx]; print('\n',tag)
 for name,phi in [('linear',xx),('cubic',xx**3),('signed2',xx*abs(xx)),('tanh',np.tanh(xx)),('sinh',np.sinh(xx)),('symmetric cubic',xx**3+3*xx),('signed saturation',xx/(1+.25*abs(xx))),('x0 cubic',xx[:,0,None]**3)]:
  pr('1,r, transform weighted '+name,np.c_[np.ones(N),rr,phi@w] if phi.shape[1]==4 and name!='linear' else np.c_[np.ones(N),rr,phi] if name=='x0 cubic' else np.c_[np.ones(N),rr]);
  if name!='linear': pr('free linear + '+name,np.c_[dd,phi]);
 for name,phi in [('sum-cubes',np.sum(xx**3,1)),('prod-all',np.prod(xx,1)),('r cubed',rr**3),('lin cubed',(rr-b[2])**3),('lin norm squared',(rr-b[2])*np.sum(xx*xx,1)),('r norm squared',rr*np.sum(xx*xx,1))]: pr(name,np.c_[np.ones(N),rr,phi])
# Search one additive term at a time, including cube of sums/differences and powers of individual coordinates.
lib=[]
def add(name,z):
 if np.std(z)>1e-9: lib.append((name,z))
for i in range(4):
 for j in range(i+1,4):
  for s in (-2,-1,-.5,.5,1,2):
   z=x[:,i]+s*x[:,j]; nm=f'(x{i}{s:+}x{j})'
   for p in (2,3): add(f'{nm}^{p}',z**p)
   add(f'{nm}abs',z*abs(z)); add(f'{nm}rational',z/(1+z*z)); add(f'{nm}tanh',np.tanh(z))
for ids in combinations_with_replacement(range(4),3): add(str(ids),np.prod(x[:,ids],1))
for j in range(4):
 for power in [2,3,4,5]: add(f'x{j}^{power}',x[:,j]**power)
 for f,n in [(np.tanh,'tanh'),(np.sinh,'sinh'),(np.sin,'sin'),(np.abs,'abs')]: add(f'{n}x{j}',f(x[:,j]))
for base in [np.c_[np.ones(N),r],drift]:
 vals=[]
 for n,z in lib:
  a,l,m,ll,c=evaluate(np.c_[base,z]); vals.append((l,ll,a,m,n,c))
 for v in sorted(vals,key=lambda v:v[0])[:10]: print('single term',np.round(v[:4],5),v[4],np.round(v[5],5))
# Smooth shared coordinate transform with fitted mixing of linear and nonlinear terms.
for tag in ['sinh','rational','asinh','signed_power','cubic rational','weighted cube shifted']:
 def trans(xx,a):
  if tag=='sinh': return np.sinh(a*xx)/a
  if tag=='rational': return xx/(1+a*xx**2)
  if tag=='asinh': return np.arcsinh(a*xx)/a
  if tag=='signed_power': return np.sign(xx)*abs(xx)**(1+a)
  if tag=='cubic rational': return xx**3/(1+a*xx**2)
  return (xx+a)**3
 def f(p,xx): return p[0]+p[1]*(xx@w)+p[2]*(trans(xx,p[3])@w)
 lo=-3 if tag=='weighted cube shifted' else .001; hi=3
 def fit(ok): return least_squares(lambda p:f(p,x[ok])-target[ok],[.27,1,.1,.4],bounds=([-np.inf]*3+[lo],[np.inf]*3+[hi]),max_nfev=500).x
 es=np.array([f(fit(np.arange(N)!=i),x[i:i+1])[0]-target[i] for i in range(N)]); p=fit(np.ones(N,bool)); print('nonlinear',tag,np.round(p,5),np.sqrt(np.mean((f(p,x)-target)**2)),np.sqrt(np.mean(es**2)),max(abs(es)))
