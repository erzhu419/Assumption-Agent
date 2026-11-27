import contextlib,io,itertools
with contextlib.redirect_stdout(io.StringIO()):
 import chamber_probe as p
import numpy as np
from scipy.optimize import least_squares
m,g,t,Y,R,theta=p.m,p.g,p.t,p.Y,p.R,p.theta
x=m._features(g,g)[0]; past,h,F=m._specimen(g,x,p.bridge); q=F@theta[2:]; E=Y-(h[:,None]*theta[:2])@R.T
B={'t':np.r_[t,1.],'R0':R[:,0],'R1':R[:,1],'t2':np.r_[t*t,0.],'exp1':np.r_[-np.expm1(-t),1.],'log1':np.r_[np.log1p(t),1.],'sqrt1':np.r_[2*(np.sqrt(1+t)-1),1.],'rat1':np.r_[t/(1+t),1.],'rat2':np.r_[t/(1+t*t),1.]}
def fit(b):
 a=np.linalg.lstsq(b.T,E.T,rcond=None)[0].T; return np.sqrt(np.mean((a@b-E)**2)),a
print('joint curve/rate singular values',np.linalg.svd(E,compute_uv=False)[:6]); print('fixed pairs',sorted((fit(np.array([B[a],B[b]]))[0],a,b) for a,b in itertools.combinations(B,2))[:10]);
def fun(kind,z):
 if kind=='exp':return np.r_[-np.expm1(-t*z)/z,1.]
 if kind=='log':return np.r_[np.log1p(t*z)/z,1.]
 if kind=='sq':return np.r_[2*t/(np.sqrt(1+t*z)+1),1.]
 if kind=='pow':return np.r_[t/np.sqrt(1+t*z),1.]
 if kind=='rat':return np.r_[t/(1+t*z),1.]
 if kind=='rat2':return np.r_[t/(1+(t*z)**2),1.]
res=[]
for pair in itertools.combinations_with_replacement(['exp','log','sq','pow','rat','rat2','t','R0','R1','t2'],2):
 def basis(z):return np.array([B[k] if k in B else fun(k,v) for k,v in zip(pair,z)])
 def err(z):
  b=basis(z); return (fit(b)[1]@b-E).ravel()
 z=min([least_squares(err,init,bounds=(.0001,20),max_nfev=70) for init in ((.2,1.),(1.,.2),(.7,2.))],key=lambda f:f.cost).x
 b=basis(z); loss,a=fit(b);res.append((loss,pair,z,a,b))
for loss,pair,z,a,b in sorted(res,key=lambda row:row[0])[:7]:print('shared modes',round(loss,5),pair,np.round(z,4),'amp correlations with [q,h,past]',np.round(np.corrcoef(np.column_stack((a,q,h,past)).T)[:2,2:],2).tolist())
loss,pair,z,a,b=min(res,key=lambda row:row[0]); print('best amplitudes: lab, missing, q, hold, mode1, mode2');print(np.round(np.column_stack((g['lab_id'],np.isnan(g['coupon_features_raw']).any(1),q,h,a)),4))
print('diagnostic keys',list(g));print('test normalized drift quantiles',np.quantile(p.xe@theta[3:]+theta[2],[0,.1,.5,.9,1]));print('diag normalized drift quantiles',np.quantile(q,[0,.1,.5,.9,1]))
