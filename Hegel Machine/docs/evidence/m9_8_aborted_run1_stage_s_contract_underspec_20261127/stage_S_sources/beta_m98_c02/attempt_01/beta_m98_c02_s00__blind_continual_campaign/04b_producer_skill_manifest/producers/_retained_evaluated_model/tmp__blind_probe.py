from transport_probe import *
from scipy.linalg import null_space
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; cal=calibrate(s)
f,B,h=prepare(s,'row','arith',-1); nom=np.repeat([s['robin_x'],s['robin_y']],2)/np.repeat(h,2)
y=cal(s['y_obs_raw']); Z=s['Z_nuisance'];q=np.linalg.qr(Z)[0]; W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]);Y=np.column_stack([cal(p['y_pulse_raw']) for p in s['reference_tracer_pulses']]); C=null_space(np.ones((1,B.shape[1]))); w0=np.full(B.shape[1],s['total_budget']/B.shape[1]);
def fit_blind(mask,pen):
 q=np.linalg.qr(Z[mask])[0]; y0=y[mask];
 def fun(z):
  S=f(nom*np.exp(z))[0][mask]; a=y0-S@w0; X=S@C; a=a-q@(q.T@a);X=X-q@(q.T@X); x=np.linalg.lstsq(X,a,rcond=None)[0]; return a-X@x,w0+C@x
 r=least_squares(lambda z:np.r_[fun(z)[0],pen*z],np.zeros(4),bounds=(-2,2),gtol=1e-13); Sd=f(nom*np.exp(r.x))[0]; err=y-Sd@fun(r.x)[1];g=np.linalg.lstsq(Z[mask],err[mask],rcond=None)[0];return nom*np.exp(r.x),fun(r.x)[1],err-Z@g
r=least_squares(lambda b:(f(b,B@W)[0]-Y).ravel(),nom,bounds=(0,np.inf),gtol=1e-13); print('tracer wall',r.x,'pulse residual by channel',np.sqrt(np.mean(r.fun.reshape(-1,4,2)**2,axis=(0,2))))
for pen in (0,1e-4,1e-3,1e-2):
 b,w,e=fit_blind(np.ones(len(y),bool),pen); group=np.tile(np.arange(len(s['sensor_ij']))//4,len(s['mu_shifts'])); err=[np.sqrt(np.mean(fit_blind(group!=i,pen)[2][group==i]**2)) for i in np.unique(group)];print('penalty',pen,'blind wall',b,'w',np.round(w,5),'in sample',np.sqrt(np.mean(e**2)),'worst held sensor group',max(err),'external tracer',np.sqrt(np.mean((f(b,B@W)[0]-Y)**2)))
print('telemetry correlations with raw',np.round(np.corrcoef(np.c_[s['aux_manifold_telemetry'],s['y_obs_raw'].reshape(-1,4)].T)[:4,4:],3))
