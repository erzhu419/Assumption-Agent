import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(u,v,h):
 f=np.c_[np.ones(len(u)),u,h]
 return f@np.linalg.lstsq(f,v,rcond=None)[0]-v
for a in 'ab':
 p=s['panel_'+a]; u,v=p['u'],p['v']; print('PANEL',a,'affine rmse',np.mean(fit(u,v,u)**2)**.5,'diag slope/intercept',np.array([np.polyfit(u[:,i],v[:,i],1) for i in range(4)]))
for k in [1,2]:
 u,v=s['panel_c']['u'],s['panel_c']['v'+str(k)]
 print('BRANCH',k,'affine rmse',np.mean(fit(u,v,u)**2)**.5)
 for name,g in [('softsign',lambda w:w/(1+abs(w))),('cos',np.cos),('sin',np.sin),('tanh',np.tanh),('rational2',lambda w:w*w/(1+w*w)),('rational3',lambda w:w/(1+w*w)),('abs',np.abs),('exp2',lambda w:np.exp(-w*w))]:
  def res(p):return fit(u,v,g(p[0]*u+p[1:])).ravel()
  sols=[least_squares(res,[h,0,0,0,0],bounds=([.02,-3,-3,-3,-3],[8,3,3,3,3]),max_nfev=200) for h in [.4,1.3,2.3,4.]]
  m=min(sols,key=lambda r:r.cost); print(name,round(np.sqrt(np.mean(m.fun**2)),9),np.round(m.x,5))
