from chamber_probe import *
from scipy.optimize import least_squares
L=H*np.sum(theta[:2]/taus); _,xd=prior._features(g,g); D=theta[2]+xd@theta[3:]; labs=g['lab_id']; valid=np.isfinite(g['coupon_features_raw']).all(1)
def response(p, L, D):
 a,b,u,v,w,offset=p; k0=-a*np.expm1(-t/a); k1=-b*np.expm1(-t/b);
 return L[:,None]*(u*k0+(1-u)*k1) + (D+offset)[:,None]*(v*k0+w*k1+(1-v-w)*t)
p0=np.r_[taus,.9,.2,.55,0.]
for name,ds in [('diag',D),('rate implied',R-L)]:
 for mode in ['fixed','free','rounded']:
  def fit(m):
   if mode=='rounded': return p0
   def pack(x): return np.r_[taus,x] if mode=='fixed' else x
   return pack(least_squares(lambda x:(response(pack(x),L[m],ds[m])-Y[m]).ravel(),p0[2:] if mode=='fixed' else p0, max_nfev=250).x)
  for subset in [valid,np.ones(n,bool)]:
   p=fit(subset); tables=[]
   for lab in np.unique(labs):
    b=fit(subset&(labs!=lab)); m=labs==lab; e=abs(response(b,L[m],ds[m])-Y[m]); tables.append([e.mean(),e[:,-1].mean(),abs((L+ds+b[-1]-R)[m]).mean()]);
   print(name,mode,subset.sum(),'params',p.round(6),'worst lab curve/late/rate',np.max(tables,axis=0).round(6))
# Direct estimate of dynamic hold and drift kernels using observed initial rate as a nuisance proxy.
f=least_squares(lambda x:(response(x,L,R-L)-Y).ravel(),p0,max_nfev=500); print('implied fit slice mean error',np.round((response(f.x,L,R-L)-Y).mean(0),5));
