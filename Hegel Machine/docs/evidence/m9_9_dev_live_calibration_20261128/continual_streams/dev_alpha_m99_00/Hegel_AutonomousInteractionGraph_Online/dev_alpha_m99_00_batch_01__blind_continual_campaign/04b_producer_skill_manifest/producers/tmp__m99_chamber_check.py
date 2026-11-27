exec(open('/tmp/m99_joint.py').read().split('models={')[0])
d = np.column_stack((one, m._features(di)))
D = d @ th[2:]
for name,A in [('no offset',(F(h,D),)*3),('offset',(F(one,h,D),)*3),('offset + history',(F(one,h,D,hist),)*3)]:
    print(name); r=fit(A,print_co=True); rr=np.empty_like(r)
    for g in groups: rr[g]=fit(A,np.setdiff1d(np.arange(N),g))[g]
    missing=np.isnan(di['coupon_features_raw']).any(1)
    print('fit',fmt(r),'LOLab',fmt(rr),'labs',[fmt(rr[g]) for g in groups],'imputed',fmt(rr[missing]))
for name,sp in [('baseline train','train'),('baseline test','test'),('diagnostic','diag')]:
    s=ds[sp] if sp!='diag' else di
    for l in np.unique(s['lab_id']):
        a=s['ref_coupons'][s['lab_id']==l]; print(name,l,round(np.median(a.mean(0)),4),round(np.median(a.std(0)),4))
