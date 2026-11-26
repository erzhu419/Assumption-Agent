import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_ep01_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
c=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']; l=tr['lab_id']; n=len(l); t=d['eval_times']; x,ok=m._aligned(d); a,b=[m._Design(*m._physical(d,p),d['past_dt'],t,f,p['initial_memory_states']) for p,f in ((tr,x[:n]),(te,x[n:]))]; y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; j,k=np.triu_indices(len(t),1); q=t[k]-t[j]>1.5; j,k=j[q],k[q]
def slope(r): return np.median((r[:,k]-r[:,j])/(t[k]-t[j]),axis=1)
def rmse(v): return np.sqrt(np.mean(v*v))
fold=np.empty(n,int)
for lab in np.unique(l):
 for st in np.unique(tr['stratum']):
  ids=np.flatnonzero((l==lab)&(tr['stratum']==st)); fold[ids]=np.arange(len(ids))%4
pair=lambda x: np.column_stack([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])
errors={name:np.zeros(n) for name in ('affine','pairwise','rational','baseline_missing_0.95','baseline_missing_0.98')}
for f in range(4):
 fit=(fold!=f)&ok[:n]; held=fold==f; lt,w=m._fit(a,y,fit); z=a.matrix(lt)@w; r=y[:,:-1]-z[:,:-1]; ds=slope(r); delta=y[:,-1]-z[:,-1]; score=x[:n]@w[3:]; prior=m._lab_drift(tr,t,r,score,l,score,fit); remaining=ds-prior
 for name in errors:
  correction=prior.copy(); weight=np.where(ok[:n],.15,1.)
  if name.startswith('baseline_missing'): weight[~ok[:n]]=float(name.rsplit('_',1)[1]); correction[~ok[:n]]=0
  if name in ('pairwise','rational'):
   xx=x[:n]; design=pair(xx) if name=='pairwise' else np.column_stack([score/(1+.5*xx[:,i]**2)-score for i in range(xx.shape[1])]); design-=np.mean(design[fit],axis=0); ds_fit=np.clip(remaining[fit],-.035,.035); beta=np.linalg.solve(design[fit].T@design[fit]+np.eye(design.shape[1])*2,design[fit].T@ds_fit); correction+=design@beta
  errors[name][held]=(ds-weight*delta-(1-weight)*correction)[held]*t[-1]
for name,e in errors.items():
 print(name,'held-out endpoint median abs by lab complete/missing; worst missing RMSE',np.round([[np.median(abs(e[(l==lab)&mask])) for mask in (ok[:n],~ok[:n])] for lab in np.unique(l)],4),round(max(rmse(e[(l==lab)&~ok[:n]]) for lab in np.unique(l)),5))
o=m.solve_alpha_ep01(d); lt,w=m._fit(a,y,ok[:n]); base=b.matrix(lt)@w; ea=o['hazard_crossing_times']-g['te_cross']; i=int(np.argmax(abs(ea))); print('worst crossing lab/missing/error/true/pred',te['lab_id'][i],not ok[n+i],ea[i],g['te_cross'][i],o['hazard_crossing_times'][i]); print('worst crossing rate error and endpoint error', (te['instant_rate_obs']-g['te_true_rate0'])[i],(o['predicted_strain']-g['te_clean'])[i,-1]); print('missing rate innovation magnitude and noise MAD',np.round(np.sort(abs((te['instant_rate_obs']-base[:,-1])[~ok[n:]])),5),m._mad((te['instant_rate_obs']-g['te_true_rate0'])[~ok[n:]]))
