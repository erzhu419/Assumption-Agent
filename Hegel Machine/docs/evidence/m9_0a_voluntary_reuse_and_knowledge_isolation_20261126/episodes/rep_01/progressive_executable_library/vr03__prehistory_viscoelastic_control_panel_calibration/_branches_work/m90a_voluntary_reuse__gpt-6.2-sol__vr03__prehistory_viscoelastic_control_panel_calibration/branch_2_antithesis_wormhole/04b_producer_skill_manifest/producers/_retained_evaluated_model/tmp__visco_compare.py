from visco_probe import *

def extract_refit(part, taus):
    out=[]
    for trace in part['traces']:
        a=design(trace,taus); y=np.asarray(trace['stress_obs']); b=fit(a,y)
        r=y-a@b; med=np.median(r); mad=1.4826*np.median(abs(r-med))
        keep=abs(r-med) <= max(.12, 7*mad)
        out.append(np.linalg.lstsq(a[keep],y[keep],rcond=None)[0][1:])
    return np.asarray(out)

def eval_fit(z,s,c,y,q,sq,penalty):
    _,inv,n=np.unique(np.c_[c,s],axis=0,return_inverse=True,return_counts=True)
    w=1/n[inv]; w/=w.mean()
    f=np.nansum(z*w[:,None],axis=0)/np.sum(np.isfinite(z)*w[:,None],axis=0)
    x=np.where(np.isfinite(z),z,f); q=np.where(np.isfinite(q),q,f)
    a=np.c_[x,np.eye(3)[s]*penalty]; b=np.c_[q,np.eye(3)[sq]*penalty]
    m=LogisticRegression(C=10,max_iter=500,tol=1e-8).fit(a,y,sample_weight=w)
    return m.predict_proba(b)[:,1],q

for k,d in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
    co,gt=d['cohort'],d['gt']; tr,qu=co['train'],co['query']; taus=np.asarray(co['taus'])
    rt=extract_refit(tr,taus); rq=extract_refit(qu,taus)
    z,q=aligned(rt,tr),aligned(rq,qu)
    c,s,y=tr['center_id'],tr['stratum'],tr['y']; cq,sq=qu['center_id'],qu['stratum']; t=gt['Z_query']; yq=gt['y_query']
    print('cohort',k,'refit observed MAE per center/column',[[round(v,5) for v in np.nanmean(abs(q[cq==j]-t[cq==j]),axis=0)] for j in np.unique(cq)])
    print('oracle test centers',[scores(yq[cq==j],gt['prob_query'][cq==j]) for j in np.unique(cq)])
    for name,zt,qq,scale in [('aligned',z,q,1.),('stratum-shrink',z,q,.25),('raw',np.where(tr['missing_channel_mask'],np.nan,rt),np.where(qu['missing_channel_mask'],np.nan,rq),1.)]:
        p,qf=eval_fit(zt,s,c,y,qq,sq,scale)
        print(name,'query',scores(yq,p),[scores(yq[cq==j],p[cq==j]) for j in np.unique(cq)],'LOCO',[scores(y[c==j],eval_fit(zt[c!=j],s[c!=j],c[c!=j],y[c!=j],zt[c==j],s[c==j],scale)[0]) for j in np.unique(c)],'max center MAE',max(np.mean(abs(qf[cq==j]-t[cq==j])) for j in np.unique(cq)))
