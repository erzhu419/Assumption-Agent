import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_w02_batch_07_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w02_batch_08.pkl','rb'))[0]['dataset'];tr,te=d['train'],d['test'];dt,t,sy=d['past_dt'],d['eval_times'],d['yield_stress'];c=m._calibrate(d['calibration_load_cells']);c[0]=np.polyfit(d['calibration_load_cells']['cmd_stress'],d['calibration_load_cells']['true_stress'],1);f=m._features(tr,te);n=len(tr['lab_id']);a=m._inputs(tr,c,dt,sy,f[:n]);b=m._inputs(te,c,dt,sy,f[n:]);z=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']];tau,w=m._fit(a,dt,t,z);tl=np.r_[t,1.];r=z-m._design(a,dt,t,np.log(tau))@w
p=np.median(r[:,:-1]/t,1); rr=r[:,-1]; missing=~np.isfinite(tr['coupon_features_raw']).all(1);print('train missing, slope, rate summaries');
for q in (missing,~missing):
 print(int(q.sum()),np.round(np.quantile(abs(p[q]),[.5,.8,.95,1]),5),np.round(np.quantile(abs(rr[q]),[.5,.8,.95,1]),5),round(np.corrcoef(p[q],rr[q])[0,1],3))
print('test state nonzero',np.count_nonzero(te['residual_plastic_state']),'range',np.nanmin(te['residual_plastic_state']),np.nanmax(te['residual_plastic_state']));print('test repeated stress groups',len(te['stress_hold_cmd'])-len(np.unique(np.c_[te['lab_id'],te['stress_hold_cmd'],te['past_stress_cmd']],axis=0)))
scale=np.r_[np.full(len(t),.017),.012]; lab=tr['lab_id'];
def fit(ix, vary):
 sub={k:v[ix] for k,v in tr.items()}; y=z[ix]; feat=f[:n][ix]; miss=missing[ix]; valid=np.isfinite(y)
 def residual(par):
  cal=c.copy();cal[1:]+=par[11:].reshape(3,2) if vary else 0
  A=m._design(m._inputs(sub,cal,dt,sy,feat),dt,t,par[:2]);rr=y-A@par[2:11];return (rr/scale)[valid]
 out=least_squares(residual,np.r_[np.log(tau),w,np.zeros(6) if vary else []],loss='cauchy',f_scale=2.2,x_scale='jac',max_nfev=70)
 return out.x
rng=np.random.default_rng(738);fold=np.empty(n,int)
for l in range(4):
 for g in range(3):
  ii=np.flatnonzero((lab==l)&(tr['stratum']==g));rng.shuffle(ii);fold[ii]=np.arange(len(ii))%4
for vary in (False,True):
 rs=np.empty_like(z);full=fit(np.arange(n),vary); print('vary',vary,'full',np.round(np.r_[np.exp(full[:2]),full[2:]],6))
 for k in range(4):
  ix=fold!=k;jx=~ix;par=fit(np.flatnonzero(ix),vary);cal=c.copy();cal[1:]+=par[11:].reshape(3,2) if vary else 0;sub={key:v[jx] for key,v in tr.items()};rs[jx]=z[jx]-m._design(m._inputs(sub,cal,dt,sy,f[:n][jx]),dt,t,par[:2])@par[2:11]
 for l in range(4):
  q=lab==l; print(l,'curve/rate med abs',np.median(abs(rs[q,:-1])),np.median(abs(rs[q,-1])),'complete rate rmse',np.sqrt(np.mean(rs[q&~missing,-1]**2)))
 print('slope versus rate OOF',(np.corrcoef(np.median(rs[:,:-1]/t,axis=1)[missing],rs[missing,-1])[0,1]),(np.corrcoef(np.median(rs[:,:-1]/t,axis=1)[~missing],rs[~missing,-1])[0,1]))
