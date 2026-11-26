exec(open('/tmp/visco_audit.py').read().split('for i,d in enumerate(data):')[0])
for i,d in enumerate(data):
    co,gt=d['cohort'],d['gt']
    t,q=co['train'],co['query']
    X,Z=[m._aligned_moduli(s,np.asarray(co['taus'])) for s in (t,q)]
    c,s,y=(t[k] for k in ('center_id','stratum','y'))
    cq,sq=q['center_id'],q['stratum']
    yq=gt['y_query']
    mask=q['missing_channel_mask']
    err=Z-gt['Z_query']
    print('cohort',i,'observed MAE per channel',np.round(np.nanmean(abs(err),axis=0),4))
    print('train stratum counts/prevalences',[(int(j),int(k),int(((c==j)&(s==k)).sum()),round(y[(c==j)&(s==k)].mean(),3)) for j in np.unique(c) for k in np.unique(s)])
    for j in np.unique(cq):
        h=cq==j
        oracle=gt['prob_query'][h]
        print('query center',j,'oracle AUC/Brier',round(roc_auc_score(yq[h],oracle),4),round(np.mean((yq[h]-oracle)**2),4),'rows with missing moduli',int(mask[h].any(axis=1).sum()))
    # Select the nuisance-intercept structure inside each outer training set only.
    for j in np.unique(c):
        outer=c==j
        candidates=[]
        for effect in (False,True):
            metrics=[]
            for k in np.unique(c[~outer]):
                valid=c==k
                f=~(valid|outer)
                _,p,_=m._risk(X[f],y[f],c[f],s[f],X[valid],s[valid],effect)
                metrics.append((roc_auc_score(y[valid],p),np.mean((y[valid]-p)**2)))
            candidates.append((max(b for _,b in metrics),-min(a for a,_ in metrics),effect))
        effect=min(candidates)[2]
        _,p,_=m._risk(X[~outer],y[~outer],c[~outer],s[~outer],X[outer],s[outer],effect)
        print('outer center',j,'inner worst Brier and negative worst AUC',[(round(b,4),round(a,4),e) for b,a,e in candidates],'chosen',effect,'AUC/Brier',round(roc_auc_score(y[outer],p),4),round(np.mean((y[outer]-p)**2),4))
