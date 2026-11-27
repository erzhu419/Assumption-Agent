from m99_inputs import *
D=np.vstack((E,er)); obs=np.c_[y,yr]; B=np.linalg.lstsq(D,obs.T,rcond=None)[0].T
p,h,_=cg; lab=g['lab_id']; n=len(h); hist=h[:,None]-R0[:,:2]*tau
keys=('lab_id','stratum','coupon_features_raw','ref_coupons')
merge=lambda *blocks:{k:np.concatenate([b[k] for b in blocks]) for k in keys}
feat={'all':m._features(tr,merge(te,g))[-n:], 'train_diag':m._features(tr,g)[-n:], 'diag':m._features(g,g)[-n:]}
def ref_transform(refset):
    xx=g['coupon_features_raw'].copy()
    for l in np.unique(lab):
        panel=refset['ref_coupons'][refset['lab_id']==l]
        xx[lab==l]=(xx[lab==l]-np.median(panel.mean(0)))/np.median(panel.std(0))
    mu=np.nanmean(x,0)
    return np.where(np.isfinite(xx),xx,mu)
feat['train_test']=ref_transform(merge(tr,te)); feat['train']=ref_transform(tr)
print('nominal material',tau,th)
for key,xx in feat.items():
    drift=xx@th[3:]; offset=np.outer(h,th[:2]); offset=np.c_[offset,np.zeros(n)]
    for model,F,fixed in [('drift',np.c_[np.ones(n),drift],True),('linear',np.c_[np.ones(n),xx],True),('linear + memory',np.c_[np.ones(n),xx,hist],True),('linear + hold',np.c_[np.ones(n),xx,h],False),('linear + hold + memory',np.c_[np.ones(n),xx,h,hist],False)]:
        target=B-offset if fixed else B
        pred=np.zeros_like(B)
        for l in np.unique(lab):
            v=lab==l; coef=np.linalg.lstsq(F[~v],target[~v],rcond=None)[0]; pred[v]=F[v]@coef
        if fixed: pred+=offset
        res=pred@D.T-obs
        rm=np.array([[np.mean(res[lab==l,:-1]**2)**.5,np.mean(res[lab==l,-1]**2)**.5] for l in np.unique(lab)])
        fit=np.linalg.lstsq(F,target,rcond=None)[0]
        erfit=(F@fit-target)@D.T
        print(key,model,'fit',np.sqrt(np.mean(erfit**2,axis=0))[[0,5,11,12]].round(4),'worst_lab',rm.max(0).round(4),'pooled',np.array([np.mean(res[:,:-1]**2),np.mean(res[:,-1]**2)])**.5)
        if key in ('all','train_test') and model in ('drift','linear','linear + memory'): print('coef',fit.T.round(4))
