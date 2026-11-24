import pickle,numpy as np
from scipy.signal import correlate
from scipy.optimize import differential_evolution
D=pickle.load(open('/holdout_task/public_tat.pkl','rb'))
for it in D:
 s=it['scan']; X=s['signals']; sens=s['sensor_xy_mm']; fs=s['fs_hz']; pin=s['calib_pin_xy_mm']; n=len(X)
 z=np.arange(-12,13); sig=3.0; tpl=-z*np.exp(-.5*(z/sig)**2); tpl/=np.linalg.norm(tpl)
 C=np.array([correlate(x,tpl,mode='same',method='fft') for x in X.T]).T
 def score(x,y,c):
  d=np.linalg.norm(sens-[x,y],axis=1)*fs/(c*1000); idx=np.arange(16); a=np.clip(d.astype(int),0,n-1); f=d-a
  return np.abs(C[a,idx]*(1-f)+C[np.minimum(a+1,n-1),idx]*f).sum()
 def obj(v):
  x1,y1,x2,y2,c=v
  return -(score(x1,y1,c)+score(x2,y2,c))
 r=differential_evolution(obj,[(-25,25),(-25,25),(-25,25),(-25,25),(1435,1545)],tol=1e-7,popsize=14,maxiter=250,seed=4,polish=True)
 print('fit',r.x,-r.fun,'truth',it['gt'])
