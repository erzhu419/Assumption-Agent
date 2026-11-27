import numpy as np
from scipy.optimize import least_squares


def refine(m,d):
    tr,te = d['train'],d['test']
    x,miss = m._features(tr,te)
    n = len(tr['lab_id']); x = x[:n]; lab = tr['lab_id']
    checks = d['calibration_load_cells']; cl,cx,cy = map(np.asarray,(checks['lab_id'],checks['cmd_stress'],checks['true_stress']))
    cal = m._calibration(checks)
    usable = np.isfinite(cx)&np.isfinite(cy)
    sparse = []
    for j in range(4):
        k = usable & (cl==j) & (np.abs(cy-(cal[j,0]*cx+cal[j,1]))<.0015)
        if (k.sum()<3 or np.ptp(cx[k])<.25) and np.sum((lab==j)&~miss[:n])>=8:
            sparse.append((j,k))
    if not sparse:
        return cal
    dt,t = d['past_dt'],d['eval_times']; y = tr['strain_curves_obs']-tr['strain_0'][:,None]; rate = tr['instant_rate_obs']
    # Fit the shared law on labs whose instrument checks determine the calibration.
    # If most labs are sparse, keep the instrument-reconciled fit rather than
    # extrapolate a shared law from too few coupons.
    good = ~miss[:n]&~np.isin(lab,[j for j,_ in sparse])
    if good.sum()<20:
        return cal
    s = m._stresses(tr,cal)
    tau,coef = m._fit(tuple(a[good] for a in s),dt,t,x[good],y[good],rate[good])
    for j,check in sparse:
        k = (lab==j)&~miss[:n]; past,hold,mem=(a[k] for a in s)
        a,r=m._design(tau,(past,hold,mem),dt,t,x[k],d['yield_stress'],False)
        yc,yr=y[k],rate[k]; cm,rm=np.isfinite(yc),np.isfinite(yr)
        err=np.r_[(a@coef-yc)[cm],(r@coef-yr)[rm]]
        for ml,mh in [(tr['past_stress_cmd'][k],tr['stress_hold_cmd'][k]),(np.ones_like(past),np.ones_like(hold))]:
            da,dr=m._design(tau,(ml,mh,np.zeros_like(mem)),dt,t,np.zeros_like(x[k]),d['yield_stress'],False)
            deriv=np.r_[(da@coef)[cm],(dr@coef)[rm]]
            if 'cols' not in locals(): cols=[]
            cols.append(deriv)
        J=np.column_stack(cols); del cols
        check_err = cx[check]*cal[j,0]+cal[j,1]-cy[check]
        Jc = np.column_stack((cx[check],np.ones(check.sum())))
        # Reliable sparse checks anchor the affine line; the slow-memory
        # response supplies gain information where these checks lack leverage.
        def f(q):
            return np.r_[(err+J@q)/.02,(check_err+Jc@q)/.00045,q/.35]
        q=least_squares(f,np.zeros(2),loss='soft_l1',f_scale=1.,bounds=(-.35,.35)).x
        cal[j]+=q
    return cal
