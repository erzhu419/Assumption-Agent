import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_b']['u']; v=s['panel_b']['v']; rng=np.random.default_rng(771)
for j in range(v.shape[1]):
 y=v[:,j]; vals=[]
 for i in range(110):
  def fun(p): return p[0]+p[1]*np.cos(u@p[2:6]+p[6])-y
  b=least_squares(fun,np.r_[np.mean(y),np.ptp(y)/2,rng.normal(0,.6,4),rng.uniform(-1,1)],max_nfev=160)
  vals.append((np.mean(abs(b.fun)),b.x))
 vals.sort(key=lambda v:v[0]); print(j, vals[0], 'nonmin',np.quantile([a for a,b in vals],[.1,.5,.9]))
