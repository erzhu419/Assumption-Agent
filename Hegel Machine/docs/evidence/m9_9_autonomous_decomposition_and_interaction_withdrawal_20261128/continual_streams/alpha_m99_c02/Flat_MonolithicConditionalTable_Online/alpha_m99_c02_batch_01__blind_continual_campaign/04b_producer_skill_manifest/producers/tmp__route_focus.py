import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from route_m99 import *
complete=~np.isnan(q).any(1)
for name,x in X.items():
 D=th[2]+x@th[3:]
 for tag,F in [('linear',np.c_[np.ones(n),D]),('quadratic',np.c_[np.ones(n),D,D**2]),('signed saturation',np.c_[np.ones(n),D,np.tanh(D)]),('linear + branch memory',np.c_[np.ones(n),D,M])]:
  A=F[complete]; C=np.linalg.lstsq(A,Z[complete],rcond=None)[0]; E=(F@C)@T.T-Y
  worst=[]
  for a in np.unique(lab):
   fit=complete&(lab!=a); chk=complete&(lab==a); K=np.linalg.lstsq(F[fit],Z[fit],rcond=None)[0]; e=(F[chk]@K)@T.T-Y[chk]; worst.append(np.sqrt([np.mean(e[:,:-1]**2),np.mean(e[:,-1]**2)]))
  print(name,tag,'coeff',C.round(4).tolist(),'worst complete',np.max(worst,axis=0).round(5),'missing',np.sqrt([np.mean(E[~complete,:-1]**2),np.mean(E[~complete,-1]**2)]).round(5))
