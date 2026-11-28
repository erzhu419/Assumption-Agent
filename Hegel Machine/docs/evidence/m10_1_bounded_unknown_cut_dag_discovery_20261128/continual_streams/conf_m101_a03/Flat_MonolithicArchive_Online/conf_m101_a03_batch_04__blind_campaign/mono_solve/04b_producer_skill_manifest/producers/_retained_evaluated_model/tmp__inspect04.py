import pickle
from itertools import combinations_with_replacement as cr
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
I,J=np.array(list(cr(range(4),2))).T
def a(x): return np.c_[np.ones(len(x)),x]
def q(x): return x[:,I]*x[:,J]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def phi(x): return np.c_[a(x),q(x),x**3]
def calibration(p):
 u,v=p['u'],p['v']; d=phi(u); w=np.ones(len(u))
 for _ in range(30):
  z=np.sqrt(w); c=fit(d*z[:,None],v*z[:,None]); e=np.linalg.norm(v-d@c,axis=1); w=np.minimum(1,(max(np.median(e)*1.4,.003)/(e+1e-12))**2)
 mask=e<.003
 c=fit(d[mask],v[mask]); c[np.abs(c)<.00025]=0
 return c,mask
def invert(c,x):
 initial=np.linalg.solve(c[1:5].T,(x-c[0]).T).T
 result=[least_squares(lambda z: phi(z[None])[0]@c-row,guess,gtol=1e-11) for row,guess in zip(x,initial)]
 return np.array([r.x for r in result]),np.array([np.linalg.norm(r.fun) for r in result])
def panel_fit(u,y,name):
 d=a(u); b=fit(d,y)
 if name=='coltanh':
  def F(x,c):return np.tanh(a(x)@c)
  c=least_squares(lambda c: F(u,c)-y,b,max_nfev=350).x
 elif name=='scale+tanh':
  def F(x,c):return c[0]+c[1]*np.tanh(a(x)@c[2:])
  c=least_squares(lambda c: F(u,c)-y,np.r_[0,1,b],max_nfev=350).x
 elif name in ('lin+tanh','lin+sin','lin+softsign'):
  f={'lin+tanh':np.tanh,'lin+sin':np.sin,'lin+softsign':lambda z:z/np.sqrt(1+z*z)}[name]
  def F(x,c):return a(x)@c[:5]+c[5]*f(a(x)@c[6:])
  c=min((least_squares(lambda c:F(u,c)-y,np.r_[b*.3,1,b*scale],max_nfev=350) for scale in (1,2,4)),key=lambda r:r.cost).x
 elif name in ('lin+square','lin+cube','lin+rat'):
  f={'lin+square':lambda z:z*z,'lin+cube':lambda z:z**3,'lin+rat':lambda z:z/(1+z*z)}[name]
  def F(x,c):return a(x)@c[:5]+c[5]*f(a(x)@c[6:])
  c=min((least_squares(lambda c:F(u,c)-y,np.r_[b,sign,b*scale],max_nfev=350) for scale in (1,2) for sign in (-1,1)),key=lambda r:r.cost).x
 else:
  fn={'linear':a,'poly2':lambda x:np.c_[a(x),q(x)],'sep3':lambda x:np.c_[a(x),x*x,x**3],'sep4':lambda x:np.c_[a(x),x*x,x**3,x**4]}[name]
  def F(x,c):return fn(x)@c
  c=fit(fn(u),y)
 return lambda x:F(x,c),c
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=145)
 c,m=calibration(s['panel_a']); print('A coefficients and inliers',m.sum(),'quadratic pairs',list(zip(I,J))); print(c)
 train=s['train']; x=np.r_[train['x'],s['test']['x']]; z,e=invert(c,x); print('inversion',np.quantile(e,[0,.5,.9,1]),'range',z.min(0),z.max(0),'largest residual ids',np.argsort(e)[-6:]); print('first calibrated rows',z[:4]); y=np.c_[train['y'],train['traj']]
 splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]; rng=np.random.default_rng(9); ids=rng.permutation(72); splits.extend((np.setdiff1d(ids,te),te) for te in np.array_split(ids,4))
 def scores(d):
  r=[]
  for tr,te in splits:
   err=np.abs(d[te]@fit(d[tr],y[tr])-y[te]);r.append([err[:,0].mean(),err[:,1:].mean()])
  return np.array(r)
 def out(name,d):
  r=scores(d);print(name,d.shape[1],'52/20',r[0],'56/16',r[1],'random worst',r[2:].max(0))
 for k,w in [('raw',x),('inverse A',z)]:
  for name,d in [('aff',a(w)),('quad',np.c_[a(w),q(w)]),('quadratic+cubic',phi(w)),('quad+tanh',np.c_[a(w),q(w),np.tanh(w)]),('paired sums and differences',np.c_[a(w),q(w),np.tanh(w),np.tanh(w[:,I[I<J]]+w[:,J[I<J]]),np.tanh(w[:,I[I<J]]-w[:,J[I<J]])])]:out(k+' '+name,d)
 for p in ['panel_b','panel_c']:
  u=s[p]['u']; idx=np.random.default_rng(4).permutation(len(u)); tr,te=idx[:19],idx[19:]
  for key,v in s[p].items():
   if key=='u':continue
   print(p,key); best=[]
   for method in ['linear','poly2','sep3','sep4','coltanh','scale+tanh','lin+tanh','lin+sin','lin+softsign','lin+square','lin+cube','lin+rat']:
    f=[panel_fit(u[tr],v[tr,k],method)[0] for k in range(4)]; cv=np.sqrt(np.mean((np.column_stack([g(u[te]) for g in f])-v[te])**2)); g=[panel_fit(u,v[:,k],method) for k in range(4)]; err=np.sqrt(np.mean((np.column_stack([f(u) for f,_ in g])-v)**2));best.append((cv,method,g));print(method,round(cv,6),round(err,6))
   for cv,method,g in sorted(best,key=lambda it:it[0])[:2]:
    print('best',method, 'parameters',np.stack([p for _,p in g]));
    for k,w in [('raw',x),('inverse A',z)]:
     v=np.column_stack([f(w) for f,_ in g]); out(k+' '+method+' panel only',a(v)); out(k+' '+method+' mixed',np.c_[a(w),v]);out(k+' '+method+' mixed + products',np.c_[a(w),v,q(v)])
