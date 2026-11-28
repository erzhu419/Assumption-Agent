import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=150)
for p,k in [('panel_b','v'),('panel_c','v1')]:
 x=s[p]['u']; y=s[p][k]; a=np.c_[np.ones(len(x)),x]; print(p)
 for case in ['pure','affine','square','pure_shift','affine_shift']:
  def d(w,z):
   z2=z-w[4:] if len(w)>4 else z
   f=z2/(1+np.exp(w[:4])*z2*z2)
   return np.c_[np.ones(len(z)),f] if 'pure' in case else (np.c_[np.ones(len(z)),z,f,z*z] if case=='square' else np.c_[np.ones(len(z)),z,f])
  def fun(w):
   h=d(w,x); return (h@np.linalg.lstsq(h,y,rcond=None)[0]-y).ravel()
  n=8 if 'shift' in case else 4
  fit=least_squares(fun,np.zeros(n),bounds=([-4]*4+[-1]*(n-4),[4]*4+[1]*(n-4)),max_nfev=150)
  print(case,np.std(fit.fun),fit.x); h=d(fit.x,x); print(np.linalg.lstsq(h,y,rcond=None)[0]); print('fixed params loo',np.std(fit.fun.reshape(y.shape)/(1-np.sum(h*np.linalg.pinv(h).T,axis=1))[:,None]))
x=s['panel_a']['u'];y=s['panel_a']['v']; q=np.max(abs(y),1)<3; x=x[q];y=y[q];print('panel a',len(x))
for j in range(4):
 a,b=x[:,j],x[:,(j+1)%4]; v=y[:,j]; print('Channel',j); print(np.c_[a,b,v][np.argsort(a)])
