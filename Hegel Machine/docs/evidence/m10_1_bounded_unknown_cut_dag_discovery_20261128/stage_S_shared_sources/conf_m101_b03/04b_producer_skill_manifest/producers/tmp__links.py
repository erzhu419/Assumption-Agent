from modes import *
from collections import defaultdict
rng=np.random.default_rng(5)
links={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rational':lambda w:w/(1+w*w),'softabs':lambda w:np.sqrt(.01+w*w),'abs':np.abs,'cos':np.cos}
D=aff(q); D2=PolynomialFeatures(2).fit_transform(q)
for j in range(3):
 yy=Z[:,j]
 for d,F in [(1,D),(2,D2)]:
  P=np.linalg.pinv(F[:n]); R=yy[:n]-F[:n]@(P@yy[:n]); opts=[]
  for name,fn in links.items():
   def pred(b):return fn(D@b[:5])*b[5]
   def resid(b):
    h=pred(b)[:n]; return R-h+F[:n]@(P@h)
   best=(np.inf,None)
   for i in range(16):
    b=rng.normal(size=6); b[:5]*=1 if i<8 else 2; b[5]=np.std(yy)
    fit=least_squares(resid,b,max_nfev=150,bounds=(-12,12)); e=np.mean(fit.fun**2)
    if e<best[0]:best=e,fit.x
   h=pred(best[1]); pp=h+F@(P@(yy[:n]-h[:n])); opts.append((best[0]**.5,np.mean(abs(pp[n:]-yy[n:])),name,best[1],ls(F[:n],yy[:n]-h[:n])))
  for res in sorted(opts,key=lambda r:r[0])[:3]: print('link',j,d,res)
