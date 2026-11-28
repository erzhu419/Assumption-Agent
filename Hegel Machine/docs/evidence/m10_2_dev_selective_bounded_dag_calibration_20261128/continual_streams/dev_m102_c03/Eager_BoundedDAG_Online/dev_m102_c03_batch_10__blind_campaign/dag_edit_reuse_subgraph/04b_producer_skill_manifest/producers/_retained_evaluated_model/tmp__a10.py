import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=S['panel_a']; x=p['u']; y=p['v']; F=np.c_[np.ones(len(x)),x]; keep=np.linalg.norm(y-F@np.linalg.lstsq(F,y,rcond=None)[0],axis=1)<4; x=x[keep]; y=y[keep]; F=F[keep]; print('STATS',S['telemetry_vec'],len(x)); a,b=x[:,[i for i,j in combinations(range(4),2)]],x[:,[j for i,j in combinations(range(4),2)]]
for tag,fn in {'affine':lambda k:x[:,:0],'sin':lambda k:np.sin(k*x),'cos':lambda k:np.cos(k*x),'exp':lambda k:np.exp(-k*x*x),'tanh':lambda k:np.tanh(k*x),'rat':lambda k:x/(1+k*x*x),'rat2':lambda k:x*x/(1+k*x*x),'sinpair':lambda k:np.c_[np.sin(k*(a+b)),np.sin(k*(a-b))],'cospair':lambda k:np.c_[np.cos(k*(a+b)),np.cos(k*(a-b))],'pairprod':lambda k:a*b,'cyclic':lambda k:x*np.roll(x,-1,axis=1)*np.roll(x,-2,axis=1)}.items():
 def fit(k):
  A=np.c_[F,fn(k)]; c=np.linalg.lstsq(A,y,rcond=None)[0]; return np.mean((A@c-y)**2),c
 r=minimize_scalar(lambda k:fit(k)[0],bounds=(.01,4),method='bounded'); print(tag,round(r.x,6),round(r.fun**.5,7));
 if r.fun<.008 or tag=='affine': print(np.round(fit(r.x)[1],5))
