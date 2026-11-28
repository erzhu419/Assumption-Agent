import numpy as np,pickle
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; ix=np.arange(len(x)); ax=np.count_nonzero(x,axis=1)==1
A=np.c_[np.ones(len(x)),x]; w=np.ones(len(x))
for _ in range(12):
 p=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]; e=np.linalg.norm(y-A@p,axis=1); w=1/np.sqrt(1+(e/np.median(e))**2)
clean=e<6*np.median(e); print('outliers',ix[~clean], 'residual gap',np.sort(e)[-6:]); np.set_printoptions(precision=6,suppress=True)
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]; U,V=chebvander(u/1.75,13),chebvander(v/1.75,13)
 p=np.linalg.lstsq(np.c_[U[:,:12],V[:,1:12]][clean],z[clean],rcond=None)[0]; print('CH',j,'Cheb own',p[:12],'cross',p[12:]);
 def score(fit):
  err=np.zeros(len(x)); edge=np.zeros(len(x))
  for fold in range(5):
   te=clean&(ix%5==fold); err[te]=fit(clean&~te,te)-z[te]
  for k in ix[ax&clean]:
   te=ix==k; edge[k]=(fit(clean&~te,te)-z[te])[0]
  return np.array([np.linalg.norm(err[clean&~ax])/np.sqrt(sum(clean&~ax)),max(abs(err)),max(abs(edge))])
 scores=[]
 for n in (7,9,11,13):
  for m in (5,7,9,11,13):
   D=np.c_[U[:,:n+1],V[:,1:m+1]]
   for lam in (0,1e-5,1e-4,1e-3):
    pen=lam*np.diag(np.r_[np.maximum(np.arange(n+1)-4,0)**2,np.maximum(np.arange(1,m+1)-3,0)**2]);
    def fit(tr,te): return D[te]@np.linalg.lstsq(np.r_[D[tr],pen],np.r_[z[tr],np.zeros(D.shape[1])],rcond=None)[0]
    a=score(fit); scores.append((max(a),n,m,lam,a))
 print('best polynomial',sorted(scores,key=lambda s:s[0])[:5]); print('degree 11,7 (unpenalized)',next(s[-1] for s in scores if s[1:4]==(11,7,0)))
 for name in ('rat','tanh_square','shift_linear','rat_shift','sin','asym'):
  def f(p):
   b,a,k,d,g,h,l,m,q=p; own=a*np.tanh(k*u)+d*u+g*u**3
   if name=='rat': cross=(h*v+l*v*v)/(1+(m*v)**2)+q*v
   if name=='tanh_square': cross=h*np.tanh(m*v)+l*np.tanh(m*v)**2+q*v
   if name=='shift_linear': cross=h*(np.tanh(m*v+l)-np.tanh(l))+q*v
   if name=='rat_shift': cross=h*(v+l)/(1+(m*(v+l))**2)-h*l/(1+(m*l)**2)+q*v
   if name=='sin': cross=h*np.sin(m*v)+l*(np.cos(m*v)-1)+q*v
   if name=='asym': cross=h*v+l*v*v/(1+(m*v)**2)+q*np.tanh(m*v)
   return b+own+cross
  def fit(tr,te):
   p=least_squares(lambda p:(f(p)-z)[tr],[0,1.6 if j%2==0 else 1.,1.2,.1,0,.4,-.25,.65,.05],max_nfev=200).x
   return f(p)[te]
  print(name,score(fit))
