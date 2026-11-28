import numpy as np,pickle
from itertools import combinations_with_replacement as C
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(suppress=True,precision=4,linewidth=130)
u=s['panel_c']['u']; B=[()]+[j for d in (1,2,3) for j in C(range(4),d)]; F=np.array([u[:,j].prod(1) for j in B]).T
for k in ('v1','v2'):
 y=s['panel_c'][k]; b=np.linalg.lstsq(F[:,:15],y,rcond=None)[0]; e=F[:,:15]@b-y; print(k,'quadratic coefficients',b,'rmse',np.std(e,0));
 for col in range(4):
  inds=list(np.argsort(-abs(b[:,col]))); print('sparse',[(m,round(np.std(F[:,inds[:m]]@np.linalg.lstsq(F[:,inds[:m]],y[:,col],rcond=None)[0]-y[:,col]),5)) for m in [5,8,10,12]])
z=s['train']['traj']; _,sv,V=np.linalg.svd(z,full_matrices=False); t=s['eval_times']; q=V[0]/V[0,-1]; print('time',q,'sv',sv)
for f in [lambda t,a:1-np.exp(-a*t),lambda t,a:t/(1+a*t),lambda t,a:np.log1p(a*t),lambda t,a:np.arctan(a*t),lambda t,a:t/np.sqrt(1+a*t*t)]:
 out=least_squares(lambda p:f(t,*p)/f(t[-1],*p)-q,[.8],bounds=(1e-5,20)); print('time fit',out.x,out.fun)
