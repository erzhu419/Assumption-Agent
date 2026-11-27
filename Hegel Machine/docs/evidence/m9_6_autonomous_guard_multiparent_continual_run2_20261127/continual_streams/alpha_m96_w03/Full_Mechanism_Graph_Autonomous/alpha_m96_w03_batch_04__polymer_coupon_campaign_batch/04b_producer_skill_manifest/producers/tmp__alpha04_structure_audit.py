import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m96_w03_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('public_alpha_m96_w03_batch_04.pkl','rb') as f: case=pickle.load(f)[0]
d=case['dataset']; tr,te=d['train'],d['test']; x,_=s._align(tr,te); cal=s._calibrate(d['calibration_load_cells']); h=s._Hold(d,tr,x,cal); lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); yy=np.c_[tr['strain_curves_obs']-h.base[:,None],tr['instant_rate_obs']]; yp=d['yield_stress']; past=(np.abs(h.hist)>yp).any(axis=1); now=np.abs(h.stress)>yp; regime=np.where(now,2,past.astype(int)); cols=np.arange(9)
features=np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]+[x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4)]+[x[:,i]/(1+x[:,j]**2) for i in range(4) for j in range(4)])
for held in range(4):
    train=lab!=held; block={k:v[train] for k,v in tr.items()}; taus,w=s._fit(s._Hold(d,block,x[train],cal),block); a,_=h.design(taus,cols); raw=a@w-yy
    ff=features.copy(); ff-=ff[train&complete].mean(axis=0); ff/=np.maximum(ff[train&complete].std(axis=0),1e-8)
    # Fit missing-basis corrections on complete training rows only. Quadratic products span squared pairwise sums/differences.
    residual=-raw; extra=ff[:,None,:]*h.time_rate[None,:,None]; mask=(train&complete)[:,None]&np.isfinite(residual)&(np.abs(residual)<.08); ex=extra[mask]; corr=np.linalg.solve(ex.T@ex+5*np.eye(ex.shape[1]),ex.T@residual[mask]); augmented=raw+extra@corr
    def score(r,m): return tuple(round(float(np.nanmedian(np.abs(z))),6) for z in (r[m,:-1],r[m,-1]))
    print('held lab',held,'base vs added drift terms by yield regime',[(j,int(m.sum()),score(raw,m),score(augmented,m)) for j in range(3) if (m:=(lab==held)&complete&(regime==j)).any()]); m=(lab==held)&complete
    print('held lab signed onset median',float(np.nanmedian(raw[m,-1])), 'signed curve medians by horizon',np.round(np.nanmedian(raw[m,:-1],axis=0),4).tolist())
print('dataset keys',sorted(d),'training keys',sorted(tr))
