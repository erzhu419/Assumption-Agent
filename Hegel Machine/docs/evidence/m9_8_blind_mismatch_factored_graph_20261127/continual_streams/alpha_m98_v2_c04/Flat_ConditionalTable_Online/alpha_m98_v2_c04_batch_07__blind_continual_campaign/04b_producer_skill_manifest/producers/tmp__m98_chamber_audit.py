import pickle
import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/m98_arrays.npz'); globals().update({k:z[k] for k in z.files}); v=th[2]+x@th[3:]; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; G=(hold[:,None]-past@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau)))*th[:2]
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; q=it['campaign_diagnostics']; tr=it['dataset']['train']; print('aux shapes',[(k, type(it['dataset'][k]['aux_chamber_telemetry']).__name__, np.shape(it['dataset'][k]['aux_chamber_telemetry'])) for k in ['train','test'] if 'aux_chamber_telemetry' in it['dataset'][k]])
features={'hold':hold,'gap1':G[:,0],'gap2':G[:,1],'last':past[:,-1],'recent_diff':hold-past[:,-1],'history_diff':past[:,-1]-past[:,-2],'early_diff':past[:,1]-past[:,0], 'last2':past[:,-2], **{f'x{i}':u for i,u in enumerate(x.T)}}

def maps(p, mode):
 T=tau*np.exp(p[:2]); d=np.exp(p[2])*v; s=np.exp(p[3]); A=G
 f=-s*np.expm1(-t/s)
 if mode=='hill': f=t/(1+(t/s)**np.exp(p[4]))**(1/np.exp(p[4]))
 if mode=='tanh': f=s*np.tanh(t/s)
 if mode=='rational2': f=t/(1+(t/s)**2)**.5
 if mode=='exp_shape': f=-s*np.expm1(-t/s-p[4]*(t/s)**2/(1+t/s))
 if mode=='log_drift': f=s*np.log1p(t/s)
 if mode=='two_exp': f=(1-p[4])*f-p[4]*2*s*np.expm1(-t/(2*s))
 if mode=='linear_tail': f=(1-p[4])*f+p[4]*t
 if mode.startswith('d_'): d=d+p[4]*features[mode[2:]]
 if mode.startswith('sat_'): d=d/(1+p[4]**2*features[mode[4:]]**2)
 if mode.startswith('elast_'): d=d*(1+p[4]*features[mode[6:]])
 if mode=='drift_center': d=d+p[4]
 if mode=='slow_memory':
  H=tau*np.exp([0.,p[4]]); A=(hold[:,None]-past@(-np.expm1(-dt[:,None]/H)*np.exp(-later[:,None]/H)))*th[:2]
 return A@(-np.expm1(-t[None,:]/T[:,None]))+d[:,None]*f, A@(1/T)+d

def fit(mode, m):
 def obj(p):
  c,r=maps(p,mode); return np.r_[((c[m]-yc[m])/.02).ravel(),(r[m]-yr[m])/.05]
 p=np.r_[[.004,.15,.21,.06], [] if mode in ['base','tanh','rational2','log_drift'] else [.12 if mode.startswith('sat_') else 0.]]
 lb=np.full(len(p),-4.); ub=-lb
 if mode=='exp_shape': lb[4]=-.95
 return least_squares(obj,p,loss='soft_l1',f_scale=1.5,bounds=(lb,ub)).x

def evaluate(c,r):
 e=abs(c-yc); f=abs(r-yr); strata=q['stratum']; groups=lambda g:[g==u for u in np.unique(g)]
 return np.array([max(np.median(e[m]) for m in groups(labs)),max(np.median(f[m]) for m in groups(labs)),max(np.median(e[m]) for m in groups(strata)),max(np.median(f[m]) for m in groups(strata)),np.max(np.median(e,axis=0)),np.quantile(e[:,-1],.90),np.median(e),np.median(f)])
for mode in ['base','hill','tanh','rational2','exp_shape','log_drift','two_exp','linear_tail','drift_center','slow_memory']+[p+f for p in ('d_','sat_','elast_') for f in features]:
 c=np.empty_like(yc); r=np.empty_like(yr)
 for l in np.unique(labs):
  m=labs==l; cc,rr=maps(fit(mode,~m),mode); c[m]=cc[m]; r[m]=rr[m]
 vals=evaluate(c,r); print(mode,np.round(vals,5), 'coeff',fit(mode,np.ones(len(labs),bool)).round(4))
