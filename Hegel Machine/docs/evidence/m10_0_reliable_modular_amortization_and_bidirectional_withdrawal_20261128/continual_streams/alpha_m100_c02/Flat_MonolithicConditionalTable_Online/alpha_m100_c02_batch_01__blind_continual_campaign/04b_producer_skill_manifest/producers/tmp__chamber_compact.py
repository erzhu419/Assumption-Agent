exec(open('/tmp/probe.py').read().split('for name,u,x in')[0])
from scipy.optimize import least_squares
from itertools import combinations
p,h=_loads(ca,q); y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; D=_features(q,q)[:len(h)]@th[2:]; lab=q['lab_id']; Z=np.zeros((len(h),2));
for j,w in enumerate(dt): Z=np.exp(-w/tau)*Z-np.expm1(-w/tau)*p[:,j,None]
print('dt,time',dt,t,'diagnostic keys',q.keys()); print('corr',np.corrcoef(np.c_[D,h,Z].T));
for wt in (.35,1.):
 Y=np.c_[y,r*wt]; B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],wt*np.r_[1/tau,1]])
 def fit(F,m,branches=None):
  if branches is None: branches=[range(F.shape[1])]*3
  A=np.stack([(F[:,j,None]*B[None,:,k]) for k in range(3) for j in branches[k]],-1)
  b=np.linalg.lstsq(A[m].reshape(-1,A.shape[-1]),Y[m].ravel(),rcond=None)[0]; return A@b,b
 def ev(name,F,branches=None):
  e=np.zeros_like(Y); worst=[]
  for l in np.unique(lab):
   m=lab==l; e[m]=fit(F,~m,branches)[0][m]-Y[m]; worst.append(np.sqrt(np.mean(e[m]**2,axis=0)))
  pred,b=fit(F,np.ones(len(h),bool),branches); et=pred-Y; print(name,'worst',round(max(np.sqrt(np.mean(e[lab==l,:-1]**2)) for l in np.unique(lab)),6),round(np.max(worst,0)[-1]/wt,6),'fit',summary(et[:,:-1])[:2],summary(et[:,-1]/wt)[:2], 'params',np.round(b,5));
 F=np.c_[np.ones(len(h)),h,D,Z,p]; print('rate weight',wt)
 ev('HD',F[:,:3]); ev('HDZ',F[:,:5]); ev('Z matching',F,[[0,1,2,3],[0,1,2,4],[0,2]]); ev('Z slow only',F,[[0,1,2],[0,1,2,4],[0,2]]);ev('Z drift only',F,[[0,1,2],[0,1,2],[0,1,2,3,4]]);ev('HD no h drift',F,[[0,1,2],[0,1,2],[0,2]]); ev('HD w/o intercept branch',F,[[1,2],[1,2],[0,2]]);
 for j in range(6): ev('past '+str(j),F,[[0,1,2,5+j],[0,1,2,5+j],[0,2]])
