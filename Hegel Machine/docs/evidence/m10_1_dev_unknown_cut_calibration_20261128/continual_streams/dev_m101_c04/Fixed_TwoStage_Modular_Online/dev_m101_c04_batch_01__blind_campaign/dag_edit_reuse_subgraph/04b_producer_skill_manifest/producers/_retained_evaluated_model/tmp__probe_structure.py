exec(open('/tmp/probe_node01.py').read().split("for w in (u,x):")[0])
from scipy.optimize import least_squares
A=poly(x,2); Q=A[:,5:]; L=A[:,:5]
for ids in [np.arange(52),np.arange(72)]:
 D=A[ids]; P=np.linalg.pinv(D); r=y[ids]-D@(P@y[ids]); X=L[ids];
 p=np.linalg.lstsq(X,y[ids],rcond=None)[0]; p*=0.8/np.sqrt(np.mean((X@p)**2));
 for name in ('tanh','atan','erf','softsign','tanh_odd','sigmoid_quadratic','tanh_times_linear','tanh_plus_cubic'):
  from scipy.special import erf
  def feature(ix,p):
   Z=L[ix]; q=Q[ix]; h=Z@p[:5];
   if name=='tanh_odd': return np.tanh(Z[:,1:]@p[1:])[:,None]
   if name=='sigmoid_quadratic': return np.tanh(h+q@p[5:])[:,None]
   if name=='tanh_times_linear': return np.tanh(h)[:,None]*Z
   if name=='tanh_plus_cubic': return np.c_[np.tanh(h),(h-p[0])**3]
   return {'tanh':np.tanh,'atan':np.arctan,'erf':erf,'softsign':lambda h:h/np.sqrt(1+h*h)}[name](h)[:,None]
  def solve(p,ret=False):
   F=feature(ids,p); G=F-D@(P@F); w=np.linalg.lstsq(G,r,rcond=1e-9)[0]; e=G@w-r
   if ret:
    b=P@(y[ids]-F@w); pred=A@b+feature(np.arange(len(x)),p)@w
    return np.mean(abs(e)),np.mean(abs(pred[52:]-y[52:])),np.max(abs(e)),w,b
   return e
  start=np.r_[p,np.zeros(10)] if name=='sigmoid_quadratic' else p.copy()
  fit=least_squares(solve,start,max_nfev=220)
  e,v,m,w,b=solve(fit.x,True); print(len(ids),name,'mae',round(e,6),round(v,6),'max',round(m,6),'p',fit.x,'w',w,'b',b)
print('train and future projection range and calibrated test bounds:',x.min(0),x.max(0),d['z_te'].min(0),d['z_te'].max(0))
