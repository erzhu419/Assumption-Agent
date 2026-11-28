import pickle,numpy as np
from scipy.optimize import least_squares
C=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X,Y=C['true_channel_loads'],C['bridge_raw']; keep=np.max(abs(Y),1)<5; ids=np.flatnonzero(keep); X,Y=X[keep],Y[keep]; Z=np.roll(X,2,axis=1)
a=ids%4!=1
for form in ('sin','sin_tanh','sin_cubic','two_sin','sin_rat','sin2','sin_cos2'):
 def pred(p):
  k,l,m,A,B,C,D,E,F=p[:9]; gains=np.r_[1,np.exp(p[13:16])]; out=A*X+B*np.sin(k*X); cross=C*Z+D*(np.cos(l*Z)-1)
  if form=='sin_tanh': out+=E*np.tanh(m*X)
  if form=='sin_cubic': out+=E*X**3
  if form=='two_sin': out+=E*np.sin(m*X)
  if form=='sin_rat': out+=E*X/(1+(m*X)**2)
  if form=='sin2': out+=E*np.sin(m*X); cross+=F*(np.cos(m*Z)-1)
  if form=='sin_cos2': cross+=E*(np.cos(m*Z)-1)
  return p[9:13]+gains*(out+cross)
 res=[]
 for m in (.8,1.8,3.2,5.):
  p=np.r_[2.5,2,m,1.27,.32,.27,.077,.001,.001,.14,-.11,.13,-.11,-.48,.023,-.47]; r=least_squares(lambda p:(pred(p)-Y)[a].ravel(),p,max_nfev=240,loss='soft_l1',f_scale=.0005); res.append(r)
 r=min(res,key=lambda r:r.cost); e=pred(r.x)-Y; print(form,np.round(np.r_[np.sqrt(np.mean(e[a]**2)),np.sqrt(np.mean(e[~a]**2)),max(abs(e[~a]).ravel())],7),np.round(r.x,6))
