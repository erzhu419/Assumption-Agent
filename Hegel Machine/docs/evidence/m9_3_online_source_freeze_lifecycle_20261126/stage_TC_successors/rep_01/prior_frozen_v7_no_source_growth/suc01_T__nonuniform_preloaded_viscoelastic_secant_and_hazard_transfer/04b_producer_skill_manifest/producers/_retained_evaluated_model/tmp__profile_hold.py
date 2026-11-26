import pickle
import numpy as np
from scipy.optimize import least_squares


def branch(taus, a, dt, t):
    z = a['initial_memory_states'].copy()
    for j, d in enumerate(dt):
        u = -np.expm1(-d / taus)
        z += u * (a['past_stress'][:, j, None] - z)
    gap = a['stress_hold_0'][:, None] - z
    return gap[:, None, :] * (-np.expm1(-t[:, None] / taus)), gap / taus


def fit(a, dt, t):
    y = np.column_stack([a['strain_curves_obs'] - a['strain_0'][:, None], a['instant_rate_obs']])
    v = np.r_[t, 1.]
    scale = np.r_[np.full(t.size, .014), .017]
    vs = v / scale
    def profile(p):
        b, r = branch(p[:2], a, dt, t)
        m = (y - np.column_stack([b @ p[2:], r @ p[2:]])) / scale
        # A median seed prevents isolated Cauchy glitches from capturing the drift.
        c = np.median(m / vs, axis=1)
        for _ in range(18):
            res = m - c[:, None] * vs
            w = 1 / (1 + res**2)
            c = (w*m) @ vs / (w @ (vs**2))
        return (m - c[:, None]*vs).ravel(), c
    starts = ([.48, 3.4, .7, .5], [.27, 1.4, .6, .5], [.95, 6., .7, .6])
    fits = [least_squares(lambda p: profile(p)[0], p0, loss='cauchy', f_scale=1, bounds=([.2,.200001,-4,-4],[8.999999,9,4,4]), max_nfev=120) for p0 in starts]
    sol = min((q for q in fits if q.x[0] < q.x[1]), key=lambda q: q.cost)
    return sol.x, profile(sol.x)[1], [(q.x,q.cost) for q in fits], profile(sol.x)[0].reshape(y.shape)*scale

if __name__ == '__main__':
    d=pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb'))[0]['dataset']; a=d['train']; t=d['eval_times']; dt=d['past_dt']; p,c,q,res=fit(a,dt,t); print('optima:',q); print('residual absolute quantiles:',np.quantile(abs(res),[.50,.80,.95],axis=0).round(5))
    g=a['lab_id']; s=a['stratum']; x=a['coupon_features_raw'].copy(); refs=a['ref_coupons']; J=np.eye(4)[g]; S=np.eye(3)[s]; good=np.isfinite(x).all(1); cs=[]
    print('lab affine slopes, intercepts and drift residual spread')
    for j in range(4):
        m=(g==j)&good; R=refs[g==j]; offset=np.nanmean(R); gain=np.nanstd(R); x[g==j]=(x[g==j]-offset)/gain; F=np.column_stack([np.ones(m.sum()),x[m]]); w=np.linalg.lstsq(F,c[m],rcond=None)[0]; cs.append(w); print(j,'loc/scale',offset,gain,'coefs',w.round(5),'residual',np.quantile(c[m]-F@w,[.1,.5,.9]).round(5))
    for F in (np.column_stack([np.ones(len(c)),x]),np.column_stack([J,x]),np.column_stack([J,x,S[:,1:]])):
        f=F[good]; w=np.linalg.lstsq(f,c[good],rcond=None)[0]; print('reference slope',w.round(5),'residual sd',np.std(c[good]-f@w)); print('by cell mean',[(j,k,round(np.mean(c[m]-F[m]@w),5)) for j in range(4) for k in range(3) if np.any(m:=good&(g==j)&(s==k))])
    print('rate-based drift noise',np.quantile(abs(res[:,-1]),[.5,.8,.95,.99]))
