from rheo_cv import *
from scipy.stats import spearmanr
for label,q in [('bench',tr),('pilot',ch)]:
    x,past,h=prep(q); f=fit2(q,q,.004); A,B=designs(q,f[0],'z'); e,r=errs(q,f,q); e=-e; r=-r; known=np.isfinite(q['coupon_features_raw']).all(1)
    lab,s=q['lab_id'],q['stratum']; raw=q['coupon_features_raw']; tel=q.get('aux_chamber_telemetry',np.empty((len(x),0))); features=np.c_[x,h,h*h,past,past*past,q['strain_0'],tel,tel*tel]; names=['x'+str(i) for i in range(4)]+['hold','hold2']+['past'+str(i) for i in range(6)]+['past2_'+str(i) for i in range(6)]+['strain0']+['tele'+str(i) for i in range(tel.shape[1])]+['tele2_'+str(i) for i in range(tel.shape[1])];
    print(label,'shape residual, q50 abs, q90 abs',np.round(np.median(abs(e[known]),axis=0),4),np.round(np.quantile(abs(e[known]),.9,axis=0),4))
    drift=np.median(e/t,axis=1); curvature=np.median(e[:,:4]/t[:4],axis=1)-np.median(e[:,-4:]/t[-4:],axis=1); print('drift-rate agreement',np.quantile(abs(drift[known]-r[known]),[.25,.5,.9]));
    for vname,v in [('drift',drift),('rate',r),('curvature',curvature)]:
        rr=np.array([spearmanr(v[known], z[known]).statistic for z in features.T]); order=np.argsort(-abs(rr))[:8]; print(vname,[(names[i],round(rr[i],3)) for i in order]);
    for l in np.unique(lab):
        ix=(lab==l)&known; v=drift[ix]; print('lab',l,'drift/rate',np.round(np.quantile(v,[.1,.5,.9]),5),np.round(np.quantile(r[ix],[.1,.5,.9]),5), 'std raw/ref',np.round(np.std(raw[ix],axis=0),2),np.round(np.std(q['ref_coupons'][lab==l],axis=0),2));
