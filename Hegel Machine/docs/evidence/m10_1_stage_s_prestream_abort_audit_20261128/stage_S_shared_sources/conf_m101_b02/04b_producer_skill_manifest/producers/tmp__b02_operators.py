import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c'];u=p['u']
np.set_printoptions(precision=4,suppress=True)
# Panel outputs are linear mixtures of coordinatewise transforms. Profile the mixing matrix.
def fit(u,v,fn,k0,shift,linear):
 def basis(x,q):
  z=x*q[:4]+(q[4:] if shift else 0)
  a=fn(z)
  return np.column_stack((np.ones(len(x)),x,a)) if linear else np.column_stack((np.ones(len(x)),a))
 def residual(q):
  A=basis(u,q);return (A@lstsq(A,v)[0]-v).ravel()
 x0=np.r_[np.full(4,k0),np.zeros(4) if shift else []]
 q=least_squares(residual,x0,bounds=(np.r_[np.full(4,.1),np.full(4,-2) if shift else []],np.r_[np.full(4,5.),np.full(4,2) if shift else []]),max_nfev=150).x
 A=basis(u,q);c=lstsq(A,v)[0]
 return lambda x:basis(x,q)@c,q,c
fns={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rational':lambda z:z/(1+z*z),'sinh':np.sinh,'cos':np.cos,'logcosh':lambda z:np.log(np.cosh(z)),'evrat':lambda z:z*z/(1+z*z),'evexp':lambda z:-np.expm1(-z*z),'log1p':lambda z:np.log1p(z*z),'sqrt':lambda z:np.sqrt(1+z*z),'cubic':lambda z:z**3,'abs':np.abs}
best={}
for key in ('v1','v2'):
 v=p[key];out=[]
 for name,fn in fns.items():
  for shift in (False,True):
   for linear in (False,True):
    if not linear and name in ('cos','logcosh','evrat','evexp','log1p','sqrt','abs'):continue
    if name=='cubic' and shift:continue
    for k0 in (1.,2.5) if name in ('sin','cos') else (1.,):
     errs=[]
     for J in (np.arange(18,24),np.arange(6)):
      I=np.setdiff1d(np.arange(24),J);predict,*_=fit(u[I],v[I],fn,k0,shift,linear);errs.append(np.mean(abs(predict(u[J])-v[J])))
     out.append((max(errs),name,shift,linear,k0,errs))
 print(key,sorted(out)[:10]);best[key]=sorted(out)[0]
 _,name,shift,linear,k0,_=best[key];pred,q,c=fit(u,v,fns[name],k0,shift,linear);print('params',q,'coeff',c,'MAE',np.mean(abs(pred(u)-v)))
