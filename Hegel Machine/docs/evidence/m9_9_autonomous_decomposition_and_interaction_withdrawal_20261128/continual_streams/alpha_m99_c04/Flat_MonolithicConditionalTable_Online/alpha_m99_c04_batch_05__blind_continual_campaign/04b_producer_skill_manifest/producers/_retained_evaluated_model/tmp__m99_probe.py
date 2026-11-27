import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/alpha_m99_c04_batch_04_solver.py'
s=importlib.util.spec_from_file_location('old',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; d=i['campaign_diagnostics']; tr=ds['train']; te=ds['test']; dt=ds['past_dt']; t=ds['eval_times']; cal=ds['calibration_load_cells']; bridge=m._fit_bridge(cal)
raw=cal['bridge_raw']; true=cal['true_channel_loads']; b,M,terms=bridge; err=np.linalg.norm(raw-b-true@M-sum((true[:,inds].prod(1)[:,None]*v for inds,v in terms),start=np.zeros_like(raw)),axis=1); print('bridge residual quantiles',np.quantile(err,[0,.5,.8,.9,.95,1])); print('bridge nonlinear terms',terms)
x,xe=m._features(tr,te); _,xd=m._features(tr,d); trc=m._calibrate(tr,bridge); dc=m._calibrate(d,bridge); taus,theta=m._fit(trc,x,dt,t); print('material',taus,theta)
def targets(g): return g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']
y,r=targets(d); c0,r0=targets(tr); ca,ra=m._design(trc,x,dt,t,taus); print('train MAD',m._mad(ca@theta-c0),m._mad(ra@theta-r0))
def report(name,c,p):
 e=c-y; f=p-r; print(name,'MAD',m._mad(e),m._mad(f),'worst lab MedAE',max(np.median(abs(e[d['lab_id']==k])) for k in np.unique(d['lab_id'])),max(np.median(abs(f[d['lab_id']==k])) for k in np.unique(d['lab_id'])),'MAE',np.mean(abs(e)),np.mean(abs(f)))
ca,ra=m._design(dc,xd,dt,t,taus); report('baseline',ca@theta,ra@theta)
h=dc['hold_stress']; past=dc['past_stress']; drift=theta[2]+xd@theta[3:]; report('history withdrawn',*m._withdrawn_response(h,drift,t,taus,theta))
phi=np.column_stack([-np.expm1(-t[:,None]/taus),t]); psi=np.r_[1/taus,1.]; A=np.vstack([phi,psi]); coeff=np.array([least_squares(lambda u:A@u-v,np.linalg.lstsq(A,v,rcond=None)[0],loss='soft_l1',f_scale=.005).x for v in np.column_stack([y,r])]); print('per coupon temporal fit MAD',m._mad(coeff@phi.T-y),m._mad(coeff@psi-r)); age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; history=past@(-np.expm1(-dt[:,None]/taus)*np.exp(-age[:,None]/taus)); print('correlations rows [hold, histories fast/slow, drift, hold*drift, history*drift fast/slow, history^2 fast/slow]'); Z=np.column_stack([h,history,drift,h*drift,history*drift[:,None],history**2]); print(np.corrcoef(Z.T,coeff.T)[:-3,-3:]); print('first 5 [hold,histories,drift,amp0,amp1,tail]',np.column_stack([h,history,drift,coeff])[:5]);
base=np.column_stack([h[:,None]*theta[:2],np.zeros(len(h))]); print('hold-subtracted amplitudes and tail [hold,drift,a0,a1,tail]:'); print(np.round(np.column_stack([h,drift,coeff-base]),4)); print('shape conditioning',np.linalg.cond(A));
from itertools import combinations
lab=d['lab_id']; w=np.r_[np.full(len(t),1/.003),1/.004]; obs=np.column_stack([y,r]);
def fit_basis(F,name):
 D=np.einsum('ij,kb->ikjb',F,A).reshape(len(h),len(w),-1); target=obs-base@A.T; pred=np.empty_like(obs)
 for k in np.unique(lab):
  trn=lab!=k; M=(D[trn]*w[None,:,None]).reshape(-1,D.shape[-1]); v=(target[trn]*w).ravel(); p=least_squares(lambda u:M@u-v,np.linalg.lstsq(M,v,rcond=None)[0],loss='soft_l1',f_scale=2).x; pred[~trn]=D[~trn]@p+base[~trn]@A.T
 M=(D*w[None,:,None]).reshape(-1,D.shape[-1]); v=(target*w).ravel(); p=least_squares(lambda u:M@u-v,np.linalg.lstsq(M,v,rcond=None)[0],loss='soft_l1',f_scale=2).x
 report(name,pred[:,:-1],pred[:,-1]); print('coefs',np.round(p.reshape(-1,3),5).tolist())
fs={'d':drift,'hd':h*drift,'h2d':h*h*drift,'d2':drift**2,'h':h,'one':np.ones(len(h)),'mem0':history[:,0],'mem1':history[:,1],'mem0d':history[:,0]*drift,'mem1d':history[:,1]*drift}
for names in [('d',),('d','hd'),('d','hd','h2d'),('d','hd','d2'),('d','hd','one','h'),('d','hd','mem0','mem1'),('d','hd','mem0d','mem1d')]: fit_basis(np.column_stack([fs[n] for n in names]),'+'.join(names))
