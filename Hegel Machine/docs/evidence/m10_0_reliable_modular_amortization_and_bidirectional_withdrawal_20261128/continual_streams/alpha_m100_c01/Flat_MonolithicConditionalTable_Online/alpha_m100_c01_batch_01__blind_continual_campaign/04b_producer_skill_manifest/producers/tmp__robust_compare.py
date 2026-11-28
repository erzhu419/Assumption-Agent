from explore import *
import inspect
def temporal(ta): return np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]])
source=inspect.getsource(m._fit)
X=m._features(di,di)[0]; yt=tr['strain_curves_obs']-tr['strain_0'][:,None]; rt=tr['instant_rate_obs']; yd=np.c_[di['strain_curves_obs']-di['strain_0'][:,None],di['instant_rate_obs']]
def metrics(e, labs):
 return np.round(np.max([[rms(e[labs==l,:-1]),rms(e[labs==l,-1])] for l in np.unique(labs)],axis=0),6)
def getfit(loss, scale):
 g=m.__dict__.copy(); exec(source.replace("loss='soft_l1'",'loss='+repr(loss)).replace('f_scale=.008','f_scale='+repr(scale)),g); return g['_fit']
for kind,loss,s in [('old','soft_l1',.008),('soft','soft_l1',.003),('soft','soft_l1',.015),('cauchy','cauchy',.003),('cauchy','cauchy',.006),('cauchy','cauchy',.012)]:
 fit=getfit(loss,s); P,H,F=inputs[0]; labs=tr['lab_id']; ta,th=fit(P,H,F,dt,t,yt,rt); te=np.empty((len(yt),len(t)+1)); foldparams=[]
 for lab in np.unique(labs):
  keep=labs!=lab; a,b=fit(P[keep],H[keep],F[keep],dt,t,yt[keep],rt[keep]); foldparams.append(np.r_[a,b]); c,d=m._design(a,P[~keep],H[~keep],F[~keep],dt,t); te[~keep]=np.c_[c@b,d@b]-np.c_[yt[~keep],rt[~keep]]
 D=temporal(ta)[:,:2]; h=inputs[2][1]; r=th[2]+X@th[3:]; base=(h[:,None]*th[:2])@D.T; M=r[:,None,None]*D.T[None,:,:].transpose(0,2,1); labs=di['lab_id']; err=np.empty_like(yd)
 for lab in np.unique(labs):
  keep=labs!=lab; c=np.linalg.lstsq(M[keep].reshape(-1,2),(yd-base)[keep].ravel(),rcond=None)[0]; err[~keep]=base[~keep]+M[~keep]@c-yd[~keep]
 co=np.linalg.lstsq(M.reshape(-1,2),(yd-base).ravel(),rcond=None)[0]; fixed=base+M@np.array([.2,2.])-yd
 print('COMPARE',kind,s,'max rel param spread',round(np.max(abs(np.array(foldparams)/np.r_[ta,th]-1)),4),'train RMSE',metrics(te,tr['lab_id']),'train clipped',metrics(np.clip(te,-.025,.025),tr['lab_id']),'chamber coefficients',co.round(6),'chamber worst',metrics(err,labs),'nominal worst',metrics(fixed,labs),flush=True)
print('train residual absolute curve/rate quantiles',q(te[:,:-1]),q(te[:,-1]))
