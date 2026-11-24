import pickle,numpy as np
from scipy.signal import correlate
from scipy.optimize import differential_evolution
D=pickle.load(open('/holdout_task/public_tat.pkl','rb'))
for it in D:
 s=it['scan']; X=s['signals']; sens=s['sensor_xy_mm']; fs=s['fs_hz']; n=len(X)
 z=np.arange(-12,13); tpl=-z*np.exp(-.5*(z/3.)**2); tpl/=np.linalg.norm(tpl)
 C=np.array([correlate(x,tpl,mode='same',method='fft') for x in X.T]).T
 def field(x,y,c):
  d=np.linalg.norm(sens-[x,y],axis=1)*fs/(c*1000); a=np.floor(d).astype(int); f=d-a; ix=np.arange(16)
  return C[a,ix]*(1-f)+C[np.minimum(a+1,n-1),ix]*f
 def obj(v):
  x1,y1,x2,y2,c=v
  sep=np.hypot(x1-x2,y1-y2)
  if sep<8: return 1e3+(8-sep)**2*100
  # per channel sum of absolute correlations handles polarity; suppress using l2 channel-normalized evidence
  q=np.abs(field(x1,y1,c))+np.abs(field(x2,y2,c))
  return -np.sum(q)
 r=differential_evolution(obj,[(-25,25),(-25,25),(-25,25),(-25,25),(1435,1545)],tol=1e-8,popsize=18,maxiter=350,seed=3,polish=True)
 pts=np.array(r.x[:4]).reshape(2,2); pts=pts[np.argsort(pts[:,0])]
 truth=it['gt']; gt=np.array(truth['absorbers_xy_mm']); print('fit',r.x[-1],pts,'truth',gt,'errs',np.linalg.norm(pts-gt,axis=1))
