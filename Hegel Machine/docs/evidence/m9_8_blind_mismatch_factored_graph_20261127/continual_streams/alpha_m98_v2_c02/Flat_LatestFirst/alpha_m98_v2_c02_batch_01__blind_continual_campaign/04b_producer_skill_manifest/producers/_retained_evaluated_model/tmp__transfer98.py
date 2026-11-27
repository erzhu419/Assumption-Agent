exec(open('/tmp/chamber98.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=160)
def preprocess(g):
    lab=g['lab_id']; raw=g['coupon_features_raw'].copy(); ref=g['ref_coupons']; summaries={}
    for l in np.unique(lab):
        ix=lab==l; panel=ref[ix]; off=np.median(np.mean(panel,0)); scale=np.median(np.std(panel,0)); summaries[int(l)]=(int(ix.sum()),off,scale); raw[ix]=(raw[ix]-off)/(scale if scale>1e-8 else 1.)
    _,inv,cnt=np.unique(np.c_[lab,g['stratum']],axis=0,return_inverse=True,return_counts=True); wt=1/cnt[inv]
    mu=np.nansum(raw*wt[:,None],0)/np.sum(np.isfinite(raw)*wt[:,None],0)
    return raw,mu,summaries
raw,mu,summaries=preprocess(ck); rt=np.where(np.isfinite(raw),raw,mu)@w[3:]+w[2]; lab=ck['lab_id']; valid=np.isfinite(raw).all(1)&(lab!=3); delta=Y-H[:,None]*(B@w[:2]); D=rt[:,None,None]*B[None,:,:];
def gain(isfit): return np.linalg.lstsq(D[isfit].reshape(-1,2),delta[isfit].ravel(),rcond=None)[0]
k=gain(valid); q=B@k; inferred=delta@q/(q@q); print('gains, implied multiples of theta',k,k/w[:2]); print('diagnostic lab, observed drift, isolated-panel drift, missing-column indicators'); print(np.c_[lab,inferred,rt,~np.isfinite(raw)]); print('panel summary diagnostic/deployment',summaries,preprocess(te)[2]); print('imputation means diagnostic/deployment',mu,preprocess(te)[1]); print('deployment missing counts',np.isnan(te['coupon_features_raw']).sum(0))
for l in np.unique(lab[valid]):
    keep=valid&(lab!=l); test=valid&(lab==l); kf=gain(keep); err=(D@kf-delta)[test]; print('held-out lab',l,'n',test.sum(),'gains',kf,'curve/rate rmse',np.mean(err[:,:-1]**2)**.5,np.mean(err[:,-1]**2)**.5,'worst horizon',np.sqrt(np.mean(err[:,:-1]**2,axis=0)).max())
for label, rr in [('cell-weighted',rt),('unweighted',np.where(np.isfinite(raw),raw,np.nanmean(raw,axis=0))@w[3:]+w[2]),('zero',np.nan_to_num(raw)@w[3:]+w[2]),('joint',R)]:
    err=(rr[:,None]*q-delta); mask=np.isnan(raw).any(1); print(label,'missing row RMSE',np.sqrt(np.mean(err[mask]**2,1)))
