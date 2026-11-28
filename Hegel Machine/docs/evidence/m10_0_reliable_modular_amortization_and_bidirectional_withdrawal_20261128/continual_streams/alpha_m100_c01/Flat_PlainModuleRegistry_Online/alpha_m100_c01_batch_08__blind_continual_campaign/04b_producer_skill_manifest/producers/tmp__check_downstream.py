import pickle, sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import _verified_upstream as upstream
import downstream_solver as dn
import batch_solver as batch
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    p = pickle.load(f)[0]
raw = p['dataset']
d = upstream.calibrate_upstream(dict(raw, chamber_verification_trials=p['campaign_diagnostics']))
tr, te, di = [d[k] for k in ('train', 'test', 'chamber_verification_trials')]
dt, t = d['past_dt'], d['eval_times']
a = batch.solve_batch(raw)
b = dn.solve_downstream(d)
for key in a:
    np.testing.assert_array_equal(a[key], b[key])
tau, theta = a['estimated_taus'], a['hereditary_weights']
assert .05 < tau[0] < tau[1] < 25
xt, xv = dn._features(tr, te)
zero, rate = dn._chamber(te, xv, np.array([0., 1e-6]), tau, theta)
np.testing.assert_allclose(zero[:, 0], te['strain_0'], atol=1e-12, rtol=0)
np.testing.assert_allclose((zero[:, 1]-zero[:, 0])/1e-6, rate, atol=2e-5, rtol=0)
def rms(a): return np.sqrt(np.mean(a*a))
P, H, F = dn._inputs(tr, xt)
Y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
R = tr['instant_rate_obs']; L = tr['lab_id']
for l in np.unique(L):
    m = L != l
    q, w = dn._fit(P[m], H[m], F[m], dt, t, Y[m], R[m])
    D, V = dn._design(q, P[~m], H[~m], F[~m], dt, t)
    e = D@w-Y[~m]
    print('benchtop held-out lab', l, 'taus', np.round(q,5), 'max weight shift',round(np.max(abs(w-theta)),5),'curve RMSE',round(rms(e),5),'curve p90',round(np.quantile(abs(e),.9),5),'rate RMSE',round(rms(V@w-R[~m]),5))
# The diagnostics are a separate feature panel. Pairing it with an empty
# split applies the exact same standardization rules to that panel alone.
empty = {k:v[:0] for k,v in di.items()}
xd, _ = dn._features(di, empty)
L = di['lab_id']; complete = ~np.isnan(di['coupon_features_raw']).any(axis=1)
P,H,F = dn._inputs(di,xd)
Y = di['strain_curves_obs']-di['strain_0'][:,None]
R = di['instant_rate_obs']; C = np.column_stack((Y,R))
B = H[:,None]*np.r_[-np.expm1(-t[:,None]/tau)@theta[:2],np.sum(theta[:2]/tau)]
E = C-B; D = F@theta[2:]
K = np.r_[dn._drift_integral(t,dn._DRIFT),1.]
for name,m in [('complete',complete),('missing',~complete),*[(f'lab {l}',L==l) for l in np.unique(L)]]:
    ec=(B+D[:,None]*K-C)[m]; print('exact artifact chamber',name,'curve/rate RMSE',np.round([rms(ec[:,:-1]),rms(ec[:,-1])],6))
for name,init,lo,hi in [('exp',[3.5],[.09],[50]),('exp+linear',[1.4,.56],[.09,0],[25,1]),('dual',[.8,.3,8.5],[.09,0,2],[2,1,50])]:
    def kernel(q):
        if name=='exp': z=dn._drift_integral(t,(q[0],1.,q[0]))
        elif name=='exp+linear': z=dn._drift_integral(t,(q[0],1.,q[0]))*q[1]+t*(1-q[1])
        else: z=dn._drift_integral(t,q)
        return np.r_[z,1.]
    for profile in (False, True):
        def fit(m):
            def fun(q):
                k=kernel(q); e=E[m]; drift=e@k/(k@k) if profile else D[m]
                return (drift[:,None]*k-e).ravel()
            return least_squares(fun,init,bounds=(lo,hi),loss='soft_l1',f_scale=.005).x
        worst=[0.,0.]
        for l in np.unique(L):
            m=L==l; k=kernel(fit(~m if profile else ~m & complete)); err=D[m,None]*k-E[m]
            worst[0]=max(worst[0],rms(err[:,:-1])); worst[1]=max(worst[1],rms(err[complete[m],:-1]))
        print('kernel CV',name,'profile',profile,'worst lab all/complete',np.round(worst,6),'full fit',np.round(fit(np.ones(len(L),dtype=bool) if profile else complete),6))
print('isolated composition and zero-time derivative passed')
