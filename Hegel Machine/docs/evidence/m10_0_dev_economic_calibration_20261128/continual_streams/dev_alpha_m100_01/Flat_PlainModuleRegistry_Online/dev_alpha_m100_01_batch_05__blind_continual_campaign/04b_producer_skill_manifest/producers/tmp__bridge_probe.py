import numpy as np,pickle
from scipy.optimize import least_squares
x,y=(lambda c:(c['true_channel_loads'],c['bridge_raw']))(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']); good=np.max(abs(y),axis=1)<5
np.set_printoptions(precision=5,suppress=True)
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]
 for name in ['lin','tanh','tanh+lin','rational','atan','poly','shifted','full']:
  def pred(p):
   b,a,k,c,h,d,e=p
   if name=='lin': return b+a*u+c*v
   if name=='tanh': return b+a*np.tanh(k*u)+c*np.tanh(h*v)
   if name=='tanh+lin': return b+a*np.tanh(k*u)+c*np.tanh(h*v)+d*u+e*v
   if name=='rational': return b+a*u/(1+k*k*u*u)+c*v/(1+h*h*v*v)+d*u+e*v
   if name=='atan': return b+a*np.arctan(k*u)+c*np.arctan(h*v)+d*u+e*v
   if name=='poly': return b+a*u+k*u**3+c*v+h*v**3+d*u**5+e*v**5
   if name=='shifted': return b+a*np.tanh(k*u+d)+c*np.tanh(h*v+e)
   if name=='full': return b+a*np.tanh(k*u)+c*np.tanh(h*v)+d*u*v+e*u*u*v
  err=[]
  for fold in range(5):
   fit=good & (np.arange(len(x))%5!=fold); val=good & ~fit
   p=least_squares(lambda p:(pred(p)-z)[fit],[0,2,1,.6,.5,0,0],max_nfev=800).x; err.extend((pred(p)-z)[val]);
  print(j,name,'rmse/max',np.sqrt(np.mean(np.square(err))),np.max(abs(np.array(err))),'p',p)
