import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/multicenter_certified_l0.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for i,e in enumerate(pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl','rb'))):
    s=e['study']; c=s['cohort']; g=c['center_id']; t=c['stratum']; y=c['y_potency']; beta=e['gt']['beta_potency']; results=[]; errors=[]
    for k in np.unique(g):
        train=g!=k; test=~train
        a=m._align({key:val[train] for key,val in c.items()}); p=m._potency(a,t[train],y[train]); err=np.linalg.norm(p-beta)/np.linalg.norm(beta); results.append(err)
        # Allow the held-out center's own panel, but only the training centers' imputation.
        raw=c['X_raw'][test]; ctrl=c['controls'][test]; valid=np.isfinite(ctrl).sum(0)>=2; ctrl=ctrl[:,valid]; z=(raw-np.median(np.nanmean(ctrl,axis=0)))/np.median(np.nanstd(ctrl,axis=0)); cells=np.column_stack((g[train],t[train])); _,inv,count=np.unique(cells,return_inverse=True,return_counts=True,axis=0); fill=np.average(a,axis=0,weights=1/count[inv.ravel()]); z=np.where(np.isfinite(z),z,fill)
        # Estimate held-out residual location after fitting stratum intercepts on training data.
        design=np.column_stack((np.ones(np.sum(train)),t[train]==1,t[train]==2,a)); yy=y[train]; b=np.linalg.lstsq(design,yy,rcond=None)[0]
        for scale in (1.,.35,.2):
            for _ in range(60):
                w=1/np.hypot(1,(yy-design@b)/scale); v=np.linalg.lstsq(design*w[:,None],yy*w,rcond=None)[0]; diff=np.linalg.norm(v-b); b=v
                if diff<1e-9*(1+np.linalg.norm(b)): break
        r=y[test]-np.column_stack((np.ones(np.sum(test)),t[test]==1,t[test]==2,z))@b; errors.append((float(np.median(r)),float(np.median(abs(r)))))
    print('study',i,'strict LOCO worst slope',max(results),'held-out (median signed error, median absolute error)',np.round(errors,4).tolist(),flush=True)
    q=s['sparse_selection']
    for factor in (1.,2.,4.,8.):
        scaled={**q,'X':factor*q['X'],'y':factor*q['y'],'lam0':factor**2*q['lam0'],'lam2':factor**2*q['lam2']}
        try:
            a=m._certify(scaled); print('sparse units',i,factor,'leaves',len(a['subcube_leaves']),'gap',a['objective']-a['certified_lower_bound'],'normalized objective error',a['objective']/factor**2-e['gt']['exact_l0_objective'],flush=True)
        except RuntimeError as exc: print('sparse units',i,factor,str(exc),flush=True)
