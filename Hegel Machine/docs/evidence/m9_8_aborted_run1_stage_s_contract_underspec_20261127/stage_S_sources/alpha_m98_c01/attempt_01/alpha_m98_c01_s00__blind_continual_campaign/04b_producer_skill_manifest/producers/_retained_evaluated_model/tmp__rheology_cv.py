import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; R,T,P=(d[k] for k in ('train','test','chamber_verification_trials')); dt,t=d['past_dt'],d['eval_times']; bridge=m._loads(d)
def subset(a,q): return {k:v[q] for k,v in a.items()}
def rms(v): return np.sqrt(np.mean(v*v))
r_tau,r_w,r_y=m._spectrum(R,m._history(R,bridge),dt,t); rp=m._panel([R,T]); r_fill=m._fill([R,T],rp); beta0=m._regress(m._design(R,rp,r_fill),r_y,.009)
err={}; diagnostics=[]
for i in range(len(P['lab_id'])):
 a=subset(P,np.arange(len(P['lab_id']))!=i); b=subset(P,[i]); tau,w,y=m._spectrum(a,m._history(a,bridge),dt,t)
 stress=(m._branches(m._history(b,bridge),dt,t,tau)@w)[0]; target=m._observations(b)[0]; diagnostics.append(np.r_[tau,w]); tt=np.r_[t,1.]; drift=np.median((target-stress)/tt)
 for pn,refs in [('TP',[T,P]),('RT',[R,T]),('RTP',[R,T,P]),('T',[T])]:
  panel=m._panel(refs)
  for fn,rows in [('same',refs),('RT',[R,T]),('R',[R]),('T',[T])]:
   fill=m._fill(rows,panel); xa0=m._design(a,panel,fill); xb0=m._design(b,panel,fill)[0]; good=np.isfinite(a['coupon_features_raw']).all(1)
   for model in ['soft','ordinary','complete','shrink','shrink_hi','quad','lab','missing']:
    xa, xb = xa0, xb0
    if model=='soft': beta=m._regress(xa,y,.013)
    elif model=='ordinary': beta=np.linalg.lstsq(xa,y,rcond=None)[0]
    elif model=='complete': beta=m._regress(xa[good],y[good],.013)
    else:
     bb=np.r_[0.,beta0[1:]]; diag=np.r_[0.,np.repeat(1. if model=='shrink' else 3.,4)]
     if model in ('quad','lab','missing'):
      def extra(rows,x):
       if model=='quad':return np.stack([x[:,j]*x[:,k] for j in range(1,5) for k in range(j,5)],axis=1)
       if model=='lab':return rows['lab_id'][:,None]==np.unique(P['lab_id'])
       return ~np.isfinite(rows['coupon_features_raw'])
      ea=extra(a,xa); eb=extra(b,xb[None,:]); xa=np.c_[xa,ea]; xb=np.r_[xb,eb[0]]; bb=np.r_[bb,np.zeros(ea.shape[1])]; diag=np.r_[0.,np.ones(4),np.full(ea.shape[1],3.)]
     beta=least_squares(lambda z:np.r_[xa@z-y,.035/.055*diag*(z-bb)]/.035,np.linalg.lstsq(xa,y,rcond=None)[0],loss='soft_l1').x
    pred=stress+tt*(xb@beta); err.setdefault((pn,fn,model),[]).append(np.r_[pred-target,xb@beta-drift])
complete=np.isfinite(P['coupon_features_raw']).all(1); labs=P['lab_id']
for key,v in sorted(err.items(),key=lambda kv:rms(np.array(kv[1])[:,-1]))[:20]+[(('TP','same','soft'),err['TP','same','soft'])]:
 v=np.asarray(v); print(key,'drift RMS complete/missing/all/worst',*[round(rms(v[q,-1]),5) for q in (complete,~complete,slice(None))],round(max(rms(v[labs==lab,-1]) for lab in np.unique(labs)),5),'strain/rate rms',round(rms(v[:,:-2]),5),round(rms(v[:,-2]),5))
print('LOO spectra [min, median, max]',np.quantile(diagnostics,[0,.5,1],axis=0))
print('fields', {k:list(d[k]) for k in ('train','test','chamber_verification_trials')})
