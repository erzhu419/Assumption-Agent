import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset'];R,T,P=[d[k] for k in ('train','test','chamber_verification_trials')];dt,t=d['past_dt'],d['eval_times'];bridge=m._loads(d)
sub=lambda a,q:{k:v[q] for k,v in a.items()}
labs=np.unique(np.r_[R['lab_id'],T['lab_id'],P['lab_id']]); tt=np.r_[t,1.]; output={}
r_tau,r_w,r_y=m._spectrum(R,m._history(R,bridge),dt,t); rp=m._panel([R,T]); beta0=m._regress(m._design(R,rp,m._fill([R,T],rp)),r_y,.009)
for kind,folds in [('LOO',[[i] for i in range(len(P['lab_id']))]),('LOLab',[np.flatnonzero(P['lab_id']==j) for j in labs])]:
 for indices in folds:
  mask=np.ones(len(P['lab_id']),bool);mask[indices]=False;a=sub(P,mask);b=sub(P,indices)
  tau,w,y=m._spectrum(a,m._history(a,bridge),dt,t)
  obs=m._observations(b);pred=m._branches(m._history(b,bridge),dt,t,tau)@w; dr=np.median((obs-pred)/tt,axis=1)
  for refname,refs in [('TP',[T,P]),('T',[T]),('RT',[R,T])]:
   panel=m._panel(refs)
   for fillname,rows in [('R',[R]),('RT',[R,T]),('TP',[T,P])]:
    fill=m._fill(rows,panel);xa=m._design(a,panel,fill);xb=m._design(b,panel,fill)
    la=(a['lab_id'][:,None]==labs).astype(float);lb=(b['lab_id'][:,None]==labs).astype(float)
    for lam in (0,.5,2,6,20):
     for r in (0,1.0):
      A=np.c_[xa,la] if lam else xa;B=np.c_[xb,lb] if lam else xb
      base=np.r_[0.,beta0[1:],np.zeros(len(labs))] if lam else np.r_[0.,beta0[1:]]
      pen=np.r_[0.,np.full(4,r),np.full(len(labs),lam)] if lam else np.r_[0.,np.full(4,r)]
      z=np.linalg.lstsq(np.r_[A,np.diag(pen)],np.r_[y,pen*base],rcond=None)[0];mu=B@z
      output.setdefault((kind,refname,fillname,lam,r),[]).append(np.c_[b['lab_id'], np.isfinite(b['coupon_features_raw']).all(1),mu-dr,pred+mu[:,None]*tt-obs])
for kind in ('LOO','LOLab'):
 rec=[]
 for k,vs in output.items():
  if k[0]!=kind:continue
  v=np.concatenate(vs); rms=lambda x:np.sqrt(np.mean(x*x));worst=max(rms(v[v[:,0]==j,2]) for j in labs)
  rec.append((round(rms(v[:,2]),5),round(worst,5),k[1:],round(rms(v[:,3:-1]),5),round(rms(v[:,-1]),5),round(rms(v[v[:,1]==0,2]),5)))
 for rec1 in sorted(rec)[:8]:print(kind,rec1)
 k=(kind,'TP','R',0,0);v=np.concatenate(output[k]);print(kind,'TP/R unpenalized by lab',[(j,np.round(v[v[:,0]==j,2],4).tolist()) for j in labs]); print(kind,'TP/R unpenalized metrics',[x for x in rec if x[2]==k[1:]])
