import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack
exec(open('/tmp/compare_vp.py').read().split('pub=pickle')[0])


def fit_prior(design, train, prior_sigma, curve_sigma, onset_sigma=.016):
    taus, coef = m._fit(design, train)
    t = design.time
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    onset = train['instant_rate_obs']
    n, k = y.shape
    # Standardized independent curve and onset observations; the last block
    # estimates coupon drift deviations not explained by the panel model.
    obs = np.r_[(y/curve_sigma).ravel(), onset/onset_sigma, np.zeros(n)]
    a = csr_matrix((np.r_[np.tile(t/curve_sigma,n), np.full(n,1/onset_sigma), np.full(n,1/prior_sigma)], (np.arange(n*(k+2)), np.r_[np.repeat(np.arange(n),k), np.arange(n), np.arange(n)])),shape=(n*(k+2),n))
    def un(q): return np.array([q[0], q[0]+(8-q[0])*q[1]])
    def regular(q):
        c,r = design.matrices(un(q))
        return np.vstack(((c/curve_sigma).reshape(-1,9), r/onset_sigma, np.zeros((n,9))))
    q = np.r_[taus[0], (taus[1]-taus[0])/(8-taus[0]), coef]
    def fun(p): return regular(p)@p[2:11] + a@p[11:] - obs
    def jac(p):
        b = np.zeros((len(obs),11)); b[:,2:] = regular(p)
        for j in range(2):
            delta = np.zeros(11); delta[j] = 1e-5
            b[:,j] = ((regular(p[:11]+delta)-regular(p[:11]-delta))@p[2:11])/(2e-5)
        return hstack((csr_matrix(b),a),format='csr')
    def loss(z):
        result = np.stack((z,np.ones_like(z),np.zeros_like(z)))
        for sl,c in [(slice(None,n*k),1.0),(slice(n*(k+1),None),4.0)]:
            u = z[sl]/c
            result[:,sl] = np.stack((c*np.log1p(u),1/(1+u),-1/(c*(1+u)**2)))
        return result
    res=least_squares(fun,np.r_[q,onset-design.matrices(taus)[1]@coef],jac=jac,loss=loss,bounds=(np.r_[.2,1e-6,np.full(9+n,-np.inf)],np.r_[8-1e-6,1,np.full(9+n,np.inf)]),x_scale='jac',max_nfev=60,tr_options={'atol':1e-9,'btol':1e-9})
    return un(res.x), res.x[2:11], np.median(abs(res.x[11:]))

pub=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))[0]; d,g=pub['dataset'],pub['gt']; t,dt,y=d['eval_times'],d['past_dt'],d['yield_stress']
tr=d['train']; n=len(tr['lab_id']); fold=np.zeros(n,int); rng=np.random.default_rng(902)
for l in range(4):
    for j in range(3):
        inds=np.flatnonzero((tr['lab_id']==l)&(tr['stratum']==j)); fold[rng.permutation(inds)]=np.arange(len(inds))%4
for split in range(-1,8):
    mask=(fold==split) if split<4 else (tr['lab_id']==split-4)
    train,test=(tr,d['test']) if split<0 else (subset(tr,~mask),subset(tr,mask)); x,xt=m._align(train,test)
    D=m._Design(train,x,dt,t,y); T=m._Design(test,xt,dt,t,y)
    truth=g['te_clean'] if split<0 else test['strain_curves_obs']; over=(np.max(abs(test['past_stress']),1)>y)|(abs(test['stress_hold_0'])>y); miss=np.isnan(test['coupon_features_raw']).any(1)
    for ps,cs in [(0.018,.025),(.032,.025),(.032,.045),(.065,.045)]:
        taus,coef,drift=fit_prior(D,train,ps,cs); c,r=T.matrices(taus)
        residual=test['instant_rate_obs']-r@coef
        # Redescending shrinkage: large onset discrepancies indicate inadequate coupon prior, not necessarily noisy onset.
        adjusted=residual*(1-(.016**2/(ps**2+.016**2))/(1+(residual/(2*ps))**2)); adjusted[miss]=residual[miss]
        no=c@coef+test['strain_0'][:,None]+residual[:,None]*t; pred=no+(adjusted-residual)[:,None]*t
        metric=np.mean if split<0 else lambda r: np.mean(np.minimum(r,.15))
        err=abs(pred-truth); raw=abs(no-truth)
        print(split,ps,cs, np.round(taus,4),np.round(coef[[5,7]],4),'drift',round(drift,4),'onset MAD',round(np.median(abs(residual[~miss])),4),'MAE',*[round(metric(err[mask]),5) for mask in (over|~over,over,~over)],'raw',round(metric(raw),5),'late',round(metric(err[:,-3:]),5),flush=True)
