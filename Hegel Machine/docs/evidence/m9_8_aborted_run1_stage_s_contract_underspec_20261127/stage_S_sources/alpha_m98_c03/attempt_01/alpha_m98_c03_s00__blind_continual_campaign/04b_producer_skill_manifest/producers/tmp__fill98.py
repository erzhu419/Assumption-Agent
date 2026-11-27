from joint98 import *
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; b,s,test=[d[k] for k in ('train','chamber_verification_trials','test')]; t,dt=d['eval_times'],d['past_dt']; cal=m._calibrate(d['calibration_load_cells']); z=np.r_[t,1.]; bp=m._panels(b,test); pp=m._panels(s); xb,xt,x=[m._standardize(a,p) for a,p in [(b,bp),(test,bp),(s,pp)]]; bf=m._impute(np.r_[xb,xt],b,test); pf=m._impute(x,s); print('bench+test fill',bf,'pilot fill',pf); print('telemetry bench/test',[(a['aux_chamber_telemetry'].shape, np.quantile(a['aux_chamber_telemetry'],[0,.5,1],axis=0).round(5)) for a in (b,test)]); print('t',t,'dt',dt)
def sub(s,k): return {name:v[k] for name,v in s.items()}
c=np.isfinite(x).all(1); ix=np.where(~c)[0]; j=np.where(~np.isfinite(x))[1]; print('missing feature counts',np.bincount(j,minlength=4)); tau,w,dr=m._fit_relaxation(b,m._stress(b,cal),dt,t,.012); p=np.r_[np.log(tau),w,m._fit_drift(m._design(xb,bf),dr,.006)]; bt,bw,bd=joint(b,m._stress(b,cal),dt,t,.012,m._design(xb,bf),p); p=np.r_[np.log(bt),bw,bd]; scores={}
for lab in [*np.unique(s['lab_id']),None]:
 keep=s['lab_id']!=lab if lab is not None else np.ones(len(x),bool); va=~keep if lab is not None else keep; sc=sub(s,keep&c); y=np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]; tt,ww,dd=joint(sc,m._stress(sc,cal),dt,t,.006,m._design(x[keep&c],pf),p); g=m._kernel(tt,dt,t,*m._stress(s,cal))@ww; rate=np.median((y-g)[:,z!=0]/z[z!=0],axis=1); fill_observed=(rate-m._design(x,pf)@dd)[ix]/dd[1:][j]+pf[j]; learned=bf.copy();
 for col in range(4):
  vals=fill_observed[(j==col)&keep[ix]]; learned[col]=np.median(vals) if len(vals) else bf[col]
 if lab is None: print('implied fillers lab/feat/val',np.c_[s['lab_id'][ix],j,fill_observed].round(4)); print('estimated',learned)
 for label,fill in [('pilot',pf),('bench',bf),('learned',learned)]:
  e=np.abs(g+(m._design(x,fill)@dd)[:,None]*z-y)[va]; miss=~c[va]; metric=[np.quantile(e[:,:-1],.9),e[:,-1].mean(),e[:,-2].mean(),e[miss,:-1].mean() if miss.any() else 0.]; print(lab,label,np.round(metric,5));
  if lab is not None:scores.setdefault(label,[]).append(metric)
print('worst q90 strain / rate MAE / final MAE / missing strain MAE', {k:np.max(v,axis=0).round(5) for k,v in scores.items()})
