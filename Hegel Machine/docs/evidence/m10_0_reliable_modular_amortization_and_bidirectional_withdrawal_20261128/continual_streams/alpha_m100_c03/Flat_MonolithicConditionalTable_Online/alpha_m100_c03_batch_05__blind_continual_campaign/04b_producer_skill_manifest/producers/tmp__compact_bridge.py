import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; y=c['true_channel_loads']; r=c['bridge_raw']; keep=abs(r).max(1)<5; y,r=y[keep],r[keep]
def fit(y,r,mode):
 def design(j,y,p):
  x=y[:,j]; z=y[:,(j+2)%4]; f=[np.ones(len(x)),x,np.sin(p[0]*x),z];
  if mode==0: f += [z*z,np.cos(p[0]*z)-1]
  if mode==1: f += [z*z,np.cos(p[1]*z)-1]
  if mode==2: f += [np.logaddexp(p[1]*z,-p[1]*z)-np.log(2)]
  if mode==3: f += [np.tanh(p[1]*z)**2]
  if mode==4: f += [z*z/(1+p[1]*z*z)]
  if mode==5: f += [np.cos(p[1]*z)-1]
  if mode==6: f += [np.expm1(-p[1]*z*z)]
  if mode==7: f += [z*np.tanh(p[1]*z)]
  if mode==8: f += [np.sqrt(1+p[1]*z*z)-1]
  return np.stack(f,1)
 def profile(p):
  a=[design(j,y,p) for j in range(4)]; w=[np.linalg.lstsq(v,r[:,j],rcond=None)[0] for j,v in enumerate(a)]; return np.column_stack([v@b for v,b in zip(a,w)])-r,w
 p=least_squares(lambda p:profile(p)[0].ravel(),[2.5,1.5],bounds=([2,.01],[3,5]),max_nfev=100).x
 _,w=profile(p); return p,np.array(w),lambda yy:np.column_stack([design(j,yy,p)@w[j] for j in range(4)])
if __name__=='__main__':
 for mode in range(9):
  p,w,fn=fit(y,r,mode); e=fn(y)-r; ee=np.zeros_like(e)
  for i in range(5):
   m=np.arange(len(y))%5!=i; ee[~m]=fit(y[m],r[m],mode)[2](y[~m])-r[~m]
  print(mode,p.round(6),'fit cv max',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(ee**2)),abs(ee).max(),'coef',w.round(5).tolist())
