import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path[:0]=['/holdout_task','/tmp/hegel_continual_archive']
from _verified_upstream import calibrate_upstream
import alpha_m100_c04_batch_01_solver as a
np.set_printoptions(precision=5,suppress=True,linewidth=140)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; D=p['dataset']; print('runtime keys:', list(D)); C=calibrate_upstream(dict(D,chamber_verification_trials=p['campaign_diagnostics'])); tr,te,d=[C[k] for k in ('train','test','chamber_verification_trials')]; t,dt=[C[k] for k in ('eval_times','past_dt')]; n=len(tr['lab_id']); x=a._features(tr,te); xd=a._features(tr,d)[n:]; cal=np.r_[np.ones(4),0]; train=a._inputs(tr,x[:n],cal); data=a._inputs(d,xd,cal); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; ta,th=a._fit(train,dt,t,y,r)
yt=d['strain_curves_obs']-d['strain_0'][:,None]; rt=d['instant_rate_obs']; past,h,cv=data; drift=cv@th[2:]; design=a._design(ta,data,dt,t); base=tuple(m@th for m in design); old=a._chamber_response(ta,th,h,cv,t)
def q(e): return np.round(np.percentile(abs(e),[25,50,75,90,95,100]),6)
print('taus theta',ta,th); print('train residual',*[q(m@th-o) for m,o in zip(a._design(ta,train,dt,t),(y,r))]); print('baseline residual',*[q(v-o) for v,o in zip(base,(yt,rt))]); print('archive residual',*[q(v-o) for v,o in zip(old,(yt,rt))]);
T=np.r_[t,0]; Y=np.column_stack((yt,rt));
def F(taus): return np.vstack((np.column_stack((-np.expm1(-t[:,None]/taus),t)),np.r_[1/taus,1]))
def fit(taus):
 f=F(taus); return np.linalg.lstsq(f,Y.T,rcond=None)[0].T
for tt in (ta, least_squares(lambda tt:(fit(tt)@F(tt).T-Y).ravel(),ta,bounds=(.051,24.999),loss='cauchy',f_scale=.015).x):
 f=F(tt); A=np.array([least_squares(lambda b:f@b-row,initial,loss='cauchy',f_scale=.015).x for initial,row in zip(fit(tt),Y)]); print('diag taus',tt,'temporal fit',q(A@f.T-Y));
 Z=h[:,None]-design[1][:,:2]*ta; cs=d['hold_channel_raw']; part=d['past_channel_raw']; print('A vs archive term and memory',np.column_stack((h,drift,Z,A))[:12]);
 for name,M in [('hold,d',[h,drift]),('hold,d,z0,z1',[h,drift,*Z.T]),('hold,cv,z0,z1',[h,*cv.T,*Z.T]),('hold,cv,past',[h,*cv.T,*past.T]),('channels,cv,z0,z1',[*cs.T,*cv.T,*Z.T])]:
  M=np.column_stack(M); b=np.linalg.lstsq(M,A,rcond=None)[0]; print(name,'coef',b.T,'amplitude RMS',np.sqrt(np.mean((M@b-A)**2,axis=0)))
