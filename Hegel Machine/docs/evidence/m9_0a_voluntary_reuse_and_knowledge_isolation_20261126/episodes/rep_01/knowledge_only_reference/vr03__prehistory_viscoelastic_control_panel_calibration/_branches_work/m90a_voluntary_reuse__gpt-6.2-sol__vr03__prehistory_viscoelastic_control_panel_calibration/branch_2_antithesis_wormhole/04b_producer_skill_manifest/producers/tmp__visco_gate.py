exec(open('/tmp/visco_audit.py').read().split('for i,d in enumerate(data):')[0])
from sklearn.metrics import balanced_accuracy_score

def unalign(z, t):
    raw = z.copy()
    c = t['center_id']
    for j in np.unique(c):
        h = c == j
        panel = t['controls'][h]
        spread = np.nanstd(panel, axis=0)
        ok = (np.isfinite(panel).sum(axis=0) >= 2) & (spread > 0)
        raw[h] = z[h] * np.median(spread[ok]) + np.median(np.nanmean(panel[:,ok],axis=0))
    return raw

for i,d in enumerate(data):
    co,gt=d['cohort'],d['gt']
    t,q=co['train'],co['query']
    taus=np.asarray(co['taus'])
    c,s,y=[t[k] for k in ('center_id','stratum','y')]
    x=m._aligned_moduli(t,taus)
    raw=unalign(x,t)
    for name, z in [('raw',raw),('panel',x)]:
        p=np.empty(len(y))
        slices=[]
        for j in np.unique(c):
            h=c==j
            _, p[h], _=m._risk(z[~h],y[~h],c[~h],s[~h],z[h],s[h])
            yh=y[h]; ph=p[h]>=.5
            slices.append((int(j), round(roc_auc_score(yh,p[h]),4), round(balanced_accuracy_score(yh,ph),4), round(np.mean(~ph[yh==0]),4), round(np.mean(ph[yh==1]),4)))
        print(i,name,'LOCO pooled AUROC/BA',round(roc_auc_score(y,p),4),round(balanced_accuracy_score(y,p>=.5),4),'center/AUROC/BA/specificity/sensitivity',slices)
    robust=m._robust_moduli
    m._robust_moduli=lambda a,y,_: np.linalg.lstsq(a,y,rcond=None)[0][1:]
    try:
        ols=m._aligned_moduli(q,taus)
    finally:
        m._robust_moduli=robust
    r=m.calibrate_viscoelastic_cohort(co)
    p=r['query_probabilities']; z=r['aligned_query_moduli']; obs=~q['missing_channel_mask']
    print(i,'observed query MAE OLS vs robust',np.abs(ols[obs]-gt['Z_query'][obs]).mean(),np.abs(z[obs]-gt['Z_query'][obs]).mean())
    for j in np.unique(q['center_id']):
        h=q['center_id']==j; yh=gt['y_query'][h]; ph=p[h]>=.5
        print(i,'query center',j,'BA/specificity/sensitivity',round(balanced_accuracy_score(yh,ph),4),round(np.mean(~ph[yh==0]),4),round(np.mean(ph[yh==1]),4))
