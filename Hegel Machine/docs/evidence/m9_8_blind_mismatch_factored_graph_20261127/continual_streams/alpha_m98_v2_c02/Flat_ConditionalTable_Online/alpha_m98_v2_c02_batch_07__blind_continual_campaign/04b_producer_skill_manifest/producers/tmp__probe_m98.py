import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_m98_v2_c02_batch_06_solver.py'
s=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; a,b=c['bridge_raw'],c['true_channel_loads'];
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
terms=[idx for k in range(4) for idx in combinations_with_replacement(range(4),k)]
def basis(x): return np.column_stack([np.prod(x[:,idx],axis=1) for idx in terms])
M=basis(b); good=np.max(np.abs(a),axis=1)<4*np.median(np.max(np.abs(a),axis=1)); co=np.linalg.lstsq(M[good],a[good],rcond=None)[0]; e=np.max(np.abs(M@co-a),axis=1); good=e<12*np.median(e); co=np.linalg.lstsq(M[good],a[good],rcond=None)[0];
for term,coef in zip(terms,co):
 if np.max(abs(coef))>0.001: print(term,coef.round(5))
def stress(bridge):
 y=bridge.reshape(-1,4); x=(y-co[0])@np.linalg.inv(co[1:5]);
 for _ in range(12):
  jac=np.zeros((len(x),4,4))
  for idx,v in zip(terms,co):
   for k,j in enumerate(idx): jac[:,j,:]+=np.prod(x[:,idx[:k]+idx[k+1:]],axis=1)[:,None]*v
  delta=np.linalg.solve(jac.transpose(0,2,1),(basis(x)@co-y)[...,None])[...,0]; x-=delta
  if np.max(abs(delta))<1e-10: break
 return x.sum(1).reshape(bridge.shape[:-1])
print('site keys',list(tr),list(te)); print('pkl items',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
x,_=m._standardize(tr,te); combined={k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}; _,xd=m._standardize(combined,di)
f=stress; M=m._design(tr,x,f,dt,t); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; taus,th=m._fit(M,y,r); A,B=M(taus); print('taus theta',taus,th,'train median',np.median(abs(A@th-y)),np.median(abs(B@th-r)))
h=f(di['hold_channel_raw']); q=f(di['past_channel_raw']); drift=th[2]+xd@th[3:]; tail=np.cumsum(dt[::-1])[::-1]-dt; z=q@(-np.expm1(-dt[:,None]/taus)*np.exp(-tail[:,None]/taus)); yd=di['strain_curves_obs']-di['strain_0'][:,None]; rd=di['instant_rate_obs']; R=-np.expm1(-t[:,None]/taus)*th[:2]; g=th[:2]/taus
for name,C in [('linear',t),('log',np.log1p(t)),('sat',-np.expm1(-t))]:
 for name2,W in [('keep',np.ones(2)),('reset',np.zeros(2)),('keep0',[1,0]),('keep1',[0,1])]:
  gap=h[:,None]-z*W; e=gap@R.T+drift[:,None]*C-yd; er=gap@g+drift-rd; print(name,name2,'median',np.median(abs(e)).round(6),np.median(abs(er)).round(6),'late',np.median(abs(e[:,-3:])).round(6))
for n in ['train','test']:
 a=d[n]['aux_chamber_telemetry']; print(n,'telemetry',a.mean(0),a.std(0))
from scipy.optimize import least_squares
ftr=np.c_[np.ones(len(xd)),xd];
for scheme in ['fixed','shape','memory','shape+memory','shape+memory+drift']:
 def pred(p):
  decay=p[0] if 'shape' in scheme else 1.; w=p[1:3] if 'memory' in scheme else np.ones(2); dc=drift+ftr@p[3:] if '+drift' in scheme else drift; G=h[:,None]-z*w
  return G@R.T+dc[:,None]*(-np.expm1(-decay*t)/decay),G@g+dc
 def resid(p):
  yp,rp=pred(p); return np.r_[(yp-yd).ravel(),rp-rd]
 init=np.r_[1.,1.,1.,np.zeros(5)]; fit=least_squares(resid,init,loss='cauchy',f_scale=.015) if scheme!='fixed' else None; p=fit.x if fit else init; yp,rp=pred(p); print(scheme,p.round(5),'med',np.median(abs(yp-yd)),np.median(abs(rp-rd))); 
 if fit is None:
  print('diag row rate residual',(rp-rd).round(4));print('diag row strain mean residual',np.median(yp-yd,axis=1).round(4)); print('diag drift',drift.round(3)); print('diag labs',di['lab_id'])
