import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _line(x, y, prior):
    i, j = np.triu_indices(len(x), 1)
    good = np.abs(x[j] - x[i]) > max(.025, .08 * np.ptp(x)) if len(x) else np.zeros(0, bool)
    i, j = i[good], j[good]
    if not len(i):
        return np.array([prior[0], np.median(y - prior[0]*x) if len(x) else prior[1]])
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    intercepts = y[i] - slopes*x[i]
    models = np.vstack((prior, np.column_stack((slopes, intercepts))))
    residual = np.abs(y - models[:, :1]*x - models[:, 1:])
    consensus = residual <= .0015
    count = consensus.sum(axis=1)
    # Break two-point ties using the historical calibration rather than the outliers.
    tie = np.mean(np.minimum(residual, .0015)**2, axis=1)
    tie += (count <= 2) * 1e-4 * np.sum((models - prior)**2, axis=1)
    k = np.lexsort((tie, -count))[0]
    keep = consensus[k]
    if keep.sum() >= 2 and np.ptp(x[keep]) > .025:
        return np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
    return _line(x[keep], y[keep], prior) if keep.any() else prior


def _calibration(checks):
    def unpack(c):
        return tuple(np.asarray(c[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    prior = np.tile([1., 0.], (4, 1))
    lab, cmd, true = unpack(checks)
    if any(np.unique(cmd[(lab == j) & np.isfinite(cmd)]).size < 3 for j in range(4)) or np.any(np.abs(true - cmd) > .002):
        archive = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
        if archive.exists():
            for record in json.loads(archive.read_text()):
                lab0, cmd0, true0 = unpack(record)
                for j in range(4):
                    m = (lab0 == j) & np.isfinite(cmd0) & np.isfinite(true0)
                    prior[j] = _line(cmd0[m], true0[m], prior[j])
    for j in range(4):
        m = (lab == j) & np.isfinite(cmd) & np.isfinite(true)
        prior[j] = _line(cmd[m], true[m], prior[j])
    return prior


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for j in np.unique(lab):
        m = lab == j
        panel = refs[m]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        if panel.size:
            center = np.median(np.nanmean(panel, axis=0))
            scale = np.median(np.nanstd(panel, axis=0, ddof=0))
            x[m] = (x[m] - center) / (scale or 1.)
    missing = ~np.isfinite(x)
    _, cell = np.unique(np.column_stack((lab, strata)), axis=0, return_inverse=True)
    means = []
    for k in range(cell.max() + 1):
        xx, ok = x[cell == k], ~missing[cell == k]
        denom = ok.sum(axis=0)
        means.append(np.divide(np.where(ok, xx, 0.).sum(axis=0), denom, out=np.full(x.shape[1], np.nan), where=denom > 0))
    means = np.asarray(means)
    count = np.isfinite(means).sum(axis=0)
    impute = np.divide(np.nansum(means, axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
    return np.column_stack((np.ones(len(x)), np.where(missing, impute, x))), missing.any(axis=1)


def _stresses(split, calibration):
    gain, bias = calibration[np.asarray(split['lab_id'], dtype=int)].T
    return (gain[:, None]*split['past_stress_cmd'] + bias[:, None], gain*split['stress_hold_cmd'] + bias,
            np.asarray(split['initial_memory_states']))


def _design(tau, stress, dt, t, x, yield_stress, plastic):
    past, hold, memory = stress
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = np.exp(-tail[:, None]/tau) * -np.expm1(-dt[:, None]/tau)
    delta = hold[:, None] - past @ kernel - memory*np.exp(-dt.sum()/tau)
    if plastic:
        excess = np.maximum(hold - yield_stress, 0.)
        x = np.column_stack((x, excess, excess**2))
    curve = np.concatenate((delta[:, None, :]*-np.expm1(-t[:, None]/tau), x[:, None, :]*t[None, :, None]), axis=2)
    return curve, np.column_stack((delta/tau, x))


def _fit(stress, dt, t, x, y, rate):
    past, hold, memory = stress
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    total = dt.sum()
    cm, rm = np.isfinite(y), np.isfinite(rate)
    obs = np.r_[y[cm], rate[rm]]

    def arrays(logtau):
        tau = np.exp(logtau)
        decay = np.exp(-dt[:, None]/tau)
        later = np.exp(-tail[:, None]/tau)
        long = np.exp(-total/tau)
        k = later*(1-decay)
        dk = (tail[:, None]*k - dt[:, None]*later*decay)/tau
        delta = hold[:, None] - past@k - memory*long
        dd = -past@dk - memory*long*total/tau
        future = np.exp(-t[:, None]/tau)
        rise = 1-future
        a = np.concatenate((delta[:, None, :]*rise, x[:, None, :]*t[None,:,None]), axis=2)
        da = dd[:,None,:]*rise - delta[:,None,:]*future*t[:,None]/tau
        b = np.column_stack((delta/tau,x))
        db = (dd-delta)/tau
        return np.vstack((a[cm],b[rm])), np.vstack((da[cm],db[rm]))

    def mv(a,b):
        return np.einsum('ij,j->i',a,b)
    def mad(e,axis=None):
        return 1.4826*np.nanmedian(np.abs(e-np.nanmedian(e,axis=axis,keepdims=True)),axis=axis)
    w=np.ones(len(obs))/.012
    def fun(p):
        a,_=arrays(p[:2]); return (mv(a,p[2:])-obs)*w
    def jac(p):
        a,da=arrays(p[:2]); return np.column_stack((da*p[2:4],a))*w[:,None]
    def opt(p,f):
        return least_squares(fun,p,jac=jac,loss='cauchy',f_scale=f,x_scale='jac',bounds=(np.r_[np.log([.051,.051]),np.full(x.shape[1]+2,-np.inf)],np.r_[np.log([24.99,24.99]),np.full(x.shape[1]+2,np.inf)]),max_nfev=180)
    starts=[]
    grid=(.2,.35,.55,.9,1.5,2.5,4.,6.5,10.)
    for i,a in enumerate(grid):
        for b in grid[i+1:]:
            q=np.log([a,b]); A,_=arrays(q)
            theta=np.linalg.lstsq(A,obs,rcond=None)[0]
            for _ in range(4):
                err=mv(A,theta)-obs
                u=1/np.sqrt(1+(err/max(.012,3*np.median(abs(err))))**2)
                theta=np.linalg.lstsq(A*u[:,None],obs*u,rcond=None)[0]
            e=mv(A,theta)-obs
            starts.append((np.mean(np.log1p((e/.025)**2)),np.r_[q,theta]))
    candidates=[opt(p,2.) for _,p in sorted(starts,key=lambda p:p[0])[:3]]
    p=min((v for v in candidates if v.x[0]<v.x[1]),key=lambda v:v.cost).x
    for _ in range(2):
        A,_=arrays(p[:2]); err=mv(A,p[2:])-obs
        ec=np.full(cm.shape,np.nan); ec[cm]=err[:cm.sum()]
        sig=np.maximum(.002,mad(ec,0)); rs=max(.002,mad(err[cm.sum():])) if rm.any() else .01
        w=np.r_[np.broadcast_to(1/sig,cm.shape)[cm],np.full(rm.sum(),1/rs)]
        p=opt(p,1.5).x
    return np.exp(p[:2]),p[2:]


def _refine_calibration(d, tr, x, missing, cal, dt, t, y, rate):
    lab = np.asarray(tr['lab_id'])
    checks = d['calibration_load_cells']; cl,cx,cy = map(np.asarray,(checks['lab_id'],checks['cmd_stress'],checks['true_stress']))
    usable = np.isfinite(cx)&np.isfinite(cy)
    sparse = []
    for j in range(4):
        k = usable & (cl==j) & (np.abs(cy-(cal[j,0]*cx+cal[j,1]))<.0015)
        if (k.sum()<3 or np.ptp(cx[k])<.25) and np.sum((lab==j)&~missing)>=8:
            sparse.append((j,k))
    if not sparse:
        return cal
    # Fit the shared law on labs whose instrument checks determine the calibration.
    # If most labs are sparse, keep the instrument-reconciled fit rather than
    # extrapolate a shared law from too few coupons.
    good = ~missing&~np.isin(lab,[j for j,_ in sparse])
    if good.sum()<20:
        return cal
    s = _stresses(tr,cal)
    tau,coef = _fit(tuple(a[good] for a in s),dt,t,x[good],y[good],rate[good])
    for j,check in sparse:
        k = (lab==j)&~missing; past,hold,mem=(a[k] for a in s)
        a,r=_design(tau,(past,hold,mem),dt,t,x[k],d['yield_stress'],False)
        yc,yr=y[k],rate[k]; cm,rm=np.isfinite(yc),np.isfinite(yr)
        err=np.r_[(a@coef-yc)[cm],(r@coef-yr)[rm]]
        cols = []
        for ml,mh in [(tr['past_stress_cmd'][k],tr['stress_hold_cmd'][k]),(np.ones_like(past),np.ones_like(hold))]:
            da,dr=_design(tau,(ml,mh,np.zeros_like(mem)),dt,t,np.zeros_like(x[k]),d['yield_stress'],False)
            deriv=np.r_[(da@coef)[cm],(dr@coef)[rm]]
            cols.append(deriv)
        J = np.column_stack(cols)
        check_err = cx[check]*cal[j,0]+cal[j,1]-cy[check]
        Jc = np.column_stack((cx[check],np.ones(check.sum())))
        # Reliable sparse checks anchor the affine line; the slow-memory
        # response supplies gain information where these checks lack leverage.
        def f(q):
            return np.r_[(err+J@q)/.02,(check_err+Jc@q)/.00045,q/.35]
        q=least_squares(f,np.zeros(2),loss='soft_l1',f_scale=1.,bounds=(-.35,.35)).x
        cal[j]+=q
    return cal


def _allocation(s, y, ridge, cap, budget):
    n = s.shape[1]
    h = s.T @ s + ridge*np.eye(n)
    c = s.T @ y
    w = np.full(n, budget/n)
    bound = np.zeros(n, dtype=int)
    if budget == 0:
        bound[:] = -1
    elif budget == n*cap:
        bound[:] = 1
    # Solve exactly on the current face; hit the next bound or release a
    # constraint with the wrong multiplier sign.
    for _ in range(12*n):
        free = bound == 0
        g = h @ w - c
        if free.any():
            f = h[np.ix_(free, free)]
            ones = np.ones(f.shape[0])
            z, v = np.linalg.solve(f, np.column_stack((-g[free], ones))).T
            nu = -z.sum()/v.sum()
            p = np.zeros(n)
            p[free] = z + nu*v
            if np.max(np.abs(p)) > 2e-13:
                steps = np.full(n, np.inf)
                up, down = p > 2e-13, p < -2e-13
                steps[up] = (cap - w[up])/p[up]
                steps[down] = -w[down]/p[down]
                k = np.argmin(steps)
                w += min(1., steps[k])*p
                if steps[k] < 1.:
                    bound[k] = 1 if up[k] else -1
                    w[k] = cap if up[k] else 0.
                    continue
                g = h @ w - c
        else:
            lo = np.max(g[bound == 1], initial=-np.inf)
            hi = np.min(g[bound == -1], initial=np.inf)
            if lo <= hi:
                nu = (lo + hi)/2 if np.isfinite(lo) and np.isfinite(hi) else (lo if np.isfinite(lo) else hi)
                break
            bound[np.argmax(np.where(bound == 1, g, -np.inf))] = 0
            bound[np.argmin(np.where(bound == -1, g, np.inf))] = 0
            continue
        violation = bound*(g - nu)
        k = np.argmax(violation)
        if violation[k] <= 2e-11:
            break
        bound[k] = 0
    w = np.clip(w, 0., cap)
    res = s @ w - y
    g = s.T @ res + ridge*w
    primal = .5*(res @ res + ridge*(w @ w))
    reduced = g - nu
    dual = primal - g @ w + nu*budget + cap*np.minimum(reduced, 0.).sum()
    return w, dict(weights=w, nu_sum=float(nu), lam_lower=np.maximum(reduced, 0.), lam_upper=np.maximum(-reduced, 0.),
                   primal_objective=float(primal), dual_lower_bound=float(min(primal, dual)))


def solve_alpha_m96_w01_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x, missing = _features(train, test)
    n = len(train['lab_id'])
    calibration = _calibration(dataset['calibration_load_cells'])
    dt, t = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    y = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    rate = np.asarray(train['instant_rate_obs'])
    keep = ~missing[:n] if (~missing[:n]).sum() >= 15 else np.ones(n, dtype=bool)
    yld = dataset['yield_stress']
    calibration = _refine_calibration(dataset, train, x[:n], missing[:n], calibration, dt, t, y, rate)
    train_stress = _stresses(train, calibration)
    tau, coef = _fit(tuple(v[keep] for v in train_stress), dt, t, x[:n][keep], y[keep], rate[keep])
    a, r = _design(tau, _stresses(test, calibration), dt, t, x[n:], yld, False)
    prediction = np.asarray(test['strain_0'])[:, None] + np.einsum('ntp,p->nt', a, coef)
    rate_obs = np.asarray(test['instant_rate_obs'])
    update = missing[n:] & np.isfinite(rate_obs)
    prediction[update] += (rate_obs[update] - r[update] @ coef)[:, None]*t
    s = dataset['S_alloc_modes']*(1 + .25*np.tanh(prediction[:, -1] - test['strain_0']))[:, None]
    z, ya = np.asarray(dataset['Z_nuisance']), np.asarray(dataset['y_alloc_obs']).ravel()
    pz = np.linalg.solve(z.T @ z + dataset['nuisance_ridge']*np.eye(z.shape[1]), z.T)
    w, cert = _allocation(s - z @ (pz @ s), ya - z @ (pz @ ya), dataset['source_ridge'], dataset['u_box'], dataset['total_budget'])
    return dict(estimated_taus=tau, predicted_strain=prediction, allocation_weights=w, kkt_certificate=cert)
