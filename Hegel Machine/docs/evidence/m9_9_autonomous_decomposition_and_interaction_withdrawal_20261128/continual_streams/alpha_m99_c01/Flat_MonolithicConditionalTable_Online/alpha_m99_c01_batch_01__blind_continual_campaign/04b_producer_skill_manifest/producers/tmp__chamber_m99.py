import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];tr=d['train'];te=d['test'];c=i['campaign_diagnostics'];dt=d['past_dt'];t=d['eval_times'];cal=d['calibration_load_cells'];raw=cal['bridge_raw'];y=cal['true_channel_loads'];b=m._bridge(cal)
print('bridge',b,'raw range',np.min(raw,0),np.max(raw,0)); print('sweeps',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in cal['single_channel_sweeps'].items()])
from itertools import combinations_with_replacement
for degree in (1,2,3):
 A=np.column_stack([np.ones(len(raw))]+[np.prod(raw[:,j],axis=1) for j in range(1,degree+1) for j in combinations_with_replacement(range(4),j)])
 pred=np.zeros_like(y)
 for q in np.array_split(np.random.default_rng(5).permutation(len(y)),5):
  z=np.setdiff1d(np.arange(len(y)),q);pred[q]=A[q]@np.linalg.lstsq(A[z],y[z],rcond=None)[0]
 e=pred-y;print('bridge OOF degree',degree,'channel rmse',np.sqrt(np.mean(e**2,0)),'stress rmse',np.sqrt(np.mean(e.sum(1)**2)))
x,xt=m._features(tr,te);p,h=m._loads(tr,b);pc,hc=m._loads(c,b);tau,beta=m._fit(x,p,h,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,beta)
_,xc=m._features(dict((k,np.concatenate([tr[k],te[k]],axis=0)) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')),c)
from scipy.linalg import lstsq
Y=np.column_stack((c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs'])); D=beta[2]+xc@beta[3:]; H=hc; P=pc; U=np.ones_like(H); n=len(H); labs=c['lab_id']; strata=c['stratum']; print('predictor ranges', np.round(np.percentile(np.column_stack((H,P,D)),[0,50,100],axis=0),3))
def kernels(ta):
 return np.vstack((np.column_stack((-np.expm1(-t[:,None]/ta),t)),np.r_[1/ta,1.]))
def build(ta,kind,xx=xc):
 D=beta[2]+xx@beta[3:]; F=kernels(ta); age=np.cumsum(dt[::-1])[::-1]-dt; weights=np.exp(-age[:,None]/ta)*-np.expm1(-dt[:,None]/ta); z=P@weights; gap=H[:,None]-z
 A0=gap*beta[:2]; zero=np.zeros_like(Y)
 if kind=='baseline': return np.empty((n,len(t)+1,0)), np.column_stack((A0,D))@F.T, []
 groups={
 'reset':([H,D,U],[H,D,U],[U]),
 'retain':([H,z[:,0],D,U],[H,z[:,1],D,U],[U]),
 'cross':([H,*z.T,D,U],[H,*z.T,D,U],[U]),
 'mean':([H,P.mean(1),D,U],[H,P.mean(1),D,U],[U]),
 'history':([H,*P.T,D,U],[H,*P.T,D,U],[U]),
 'history_feature':([H,*P.T,*xx.T,U],[H,*P.T,*xx.T,U],[U]),
 'history_taper':([H,*P.T,D,U],[H,*P.T,D,U],[D,U]),
 'history_interact':([H,*P.T,D,U,H*D,D**2],[H,*P.T,D,U,H*D,D**2],[U]),
 'gap_interact':([H,*z.T,D,U,H*D,*((z*D[:,None]).T),D**2],[H,*z.T,D,U,H*D,*((z*D[:,None]).T),D**2],[U]),
 }
 G=groups[kind]; return np.concatenate([np.column_stack(g)[:,None,:]*F[None,:,j,None] for j,g in enumerate(G)],axis=2),zero,list(map(len,G))
def run(ta,k,a,b):
 K,B,L=build(ta,k); A=K[a].reshape(len(a)*Y.shape[1],sum(L)); r=lstsq(A,(Y-B)[a].ravel())[0] if L else np.empty(0); E=K[b]@r+B[b]-Y[b]; return E,r,L
for kind in ('baseline','reset','retain','cross','mean','history','history_feature','history_taper','history_interact','gap_interact'):
 for tt in (tau,np.array([.7,4.8])) if kind in ('history','history_interact','retain','cross','gap_interact') else (tau,):
  e,q,L=run(tt,kind,np.arange(n),np.arange(n)); oo=[run(tt,kind,np.flatnonzero(labs!=l),np.flatnonzero(labs==l))[0] for l in np.unique(labs)]; rms=lambda v: np.round([np.sqrt(np.mean(v[:,:-1]**2)),np.sqrt(np.mean(v[:,-1]**2))],5); print(kind,tt,'full',rms(e),'worst lab',np.max([rms(v) for v in oo],axis=0),'coef',[np.round(b,4) for b in np.split(q,np.cumsum(L)[:-1])]);
# Regress the stable two-dimensional residual out of both histories and drift to expose its symmetry.
for tt in (tau,np.array([.7,4.8])):
 F=kernels(tt); amp=np.linalg.lstsq(F,Y.T,rcond=None)[0].T
 print('amplitudes',tt,'range',np.round(np.percentile(amp,[0,50,100],axis=0),4),'corr',np.round(np.corrcoef(np.column_stack((amp,H,P,D)).T)[:3,3:],3))
 print('drift standardization alternatives:')
 for tag,xx in [('joint',xc),('train_diag',m._features(tr,c)[1]),('diag_only',m._features(c,c)[1])]:
  K,B,L=build(tt,'history',xx); coef=lstsq(K.reshape(-1,sum(L)),Y.ravel())[0]; e=K@coef-Y; print(tag,np.round([np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e[:,-1]**2))],5))
