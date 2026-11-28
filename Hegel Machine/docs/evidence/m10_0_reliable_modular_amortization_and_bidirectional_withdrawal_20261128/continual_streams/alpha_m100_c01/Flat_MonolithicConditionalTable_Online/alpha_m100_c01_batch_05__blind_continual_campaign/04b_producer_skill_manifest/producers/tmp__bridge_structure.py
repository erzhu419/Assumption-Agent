import pickle
import numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']
X,Y=D['true_channel_loads'],D['bridge_raw']; keep=np.max(np.abs(Y),axis=1)<5; ids=np.flatnonzero(keep); X,Y=X[keep],Y[keep]; Z=np.roll(X,2,axis=1)
fit=(ids<16)|(ids%4!=0)
F={'tanh':np.tanh,'rat':lambda t:t/(1+t*t),'atan':np.arctan,'sqrt':lambda t:t/np.sqrt(1+t*t),'clip':lambda t:np.clip(t,-1,1),'sinh':lambda t:t/np.cosh(t),'poly':lambda t:t**3,'erf':None}
from scipy.special import erf
F['erf']=erf
# Common response shape with per-channel gain and offset; independent opposite-channel coefficients.
for name,f in F.items():
 for form in ('sep','sep3','sep5','shift','mix','split','clip2','quot'):
  if form=='clip2' and name!='clip': continue
  if form=='quot' and name!='rat': continue
  def pred(p):
   k,b,c,m,a,e,h,j,v=p[:9]; offsets=p[9:13]; gains=np.r_[1,np.exp(p[13:16])]; z=Z
   cross=c*z+m*np.abs(z)+a*z**3
   if form=='sep': cross=c*z+m*np.abs(z)
   if form=='sep5': cross=c*z+m*np.abs(z)+a*z**3+e*z*np.abs(z)+h*np.abs(z)**3
   if form=='split': cross=c*z+m*np.abs(z)+a*z**3+e*z*np.abs(z)+h*np.abs(z)**3+j*X**5+v*X**3
   if form=='shift': t=(X+h*z)*np.exp(k)
   elif form=='mix': t=X*np.exp(k)*(1+h*z+v*z*z)
   elif form=='split': t=X*np.exp(k+j*np.sign(X))
   else: t=X*np.exp(k)
   if form=='clip2': cross+=h*np.clip(X*np.exp(v),-1,1)
   if form=='quot': cross+=h*X*(X*X)/(1+np.exp(v)*X*X)
   return offsets+gains*(f(t)*b+X*e+cross) if form not in ('sep5','split') else offsets+gains*(f(t)*b+X*0.3+cross)
  sols=[]
  for k in (-.6,.35,.9):
   p=np.r_[k,1,.27,-.09,.005,.3,.0,0.,0.,.14,-.11,.13,-.11,-.47,.02,-.46]
   res=least_squares(lambda p:(pred(p)-Y)[fit].ravel(),p,loss='soft_l1',f_scale=.003,max_nfev=240,bounds=(np.r_[[-2],np.full(7,-np.inf),[-4],np.full(4,-np.inf),np.full(3,-2)],np.r_[[2],np.full(7,np.inf),[4],np.full(4,np.inf),np.full(3,2)])); sols.append(res)
  r=min(sols,key=lambda r:r.cost); err=pred(r.x)-Y
  print(name,form,'fit/hold/max',np.round([np.sqrt(np.mean(err[fit]**2)),np.sqrt(np.mean(err[~fit]**2)),np.max(abs(err[~fit]))],6), 'pars',np.round(r.x,5) if np.sqrt(np.mean(err[~fit]**2))<.009 else '')
