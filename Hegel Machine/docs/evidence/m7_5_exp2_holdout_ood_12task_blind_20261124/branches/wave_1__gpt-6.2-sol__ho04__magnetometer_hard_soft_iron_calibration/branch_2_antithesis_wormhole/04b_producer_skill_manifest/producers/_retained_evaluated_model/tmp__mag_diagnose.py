import numpy as np
from mag_validate import m, synthetic, errors, report

fit_current = m._fit

def prepared(s):
    raw = s['calib_m_raw']
    origin = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw-origin, axis=1))
    return (raw-origin)/scale, m._level_axes(s['calib_roll_pitch_rad']), origin, scale

def refit(p, x, axes, cutoff):
    for _ in range(4):
        distance = np.linalg.norm(m._residual(p,x,axes),axis=1)
        sigma = max(np.median(distance)/np.sqrt(2*np.log(2)),1e-6)
        keep = distance <= cutoff*sigma
        q = m.least_squares(lambda q: m._residual(q,x[keep],axes[keep]).ravel(),p,max_nfev=25).x
        if np.linalg.norm(q-p)<1e-7:
            return q
        p = q
    return p

def clean_mask(seed,n):
    rng = np.random.default_rng(seed)
    rng.normal(size=(3,3)); rng.uniform(-65,65,size=3)
    rng.uniform(-1,1,size=(n,2))
    yaw = rng.uniform(-np.pi,np.pi,size=n)
    keep = np.ones(n,dtype=bool)
    keep[np.argsort(yaw)[-round(.04*n):]] = False
    return keep

def exact_inliers(p,x,axes,keep):
    return m.least_squares(lambda q: m._residual(q,x[keep],axes[keep]).ravel(),p,max_nfev=25)

for lo, hi in [(44,108),(108,172)]:
    outcomes = {k:[] for k in ('current','trim 3.5','trim 4.5','trim 5.5','oracle inliers')}
    for i in range(lo,hi):
        s,g = synthetic(i,n=180,tilt=.5,h=12,soft=.40,spike='cluster')
        x,axes,origin,scale = prepared(s)
        p = fit_current(x,axes)
        clean = clean_mask(i,len(x))
        oracle = exact_inliers(p,x,axes,clean)
        if i == 58:
            d = np.linalg.norm(m._residual(p,x,axes),axis=1)
            keep = d <= 4*np.median(d)/np.sqrt(2*np.log(2))
            _, singular, directions = np.linalg.svd(oracle.jac,full_matrices=False)
            error = origin+scale*oracle.x[5:8]-g['b_hard']
            mode = directions[-1,5:8]
            print('seed 58 discarded clean, retained spikes, condition, bias alignment with weakest bias direction:',(~keep&clean).sum(),(keep&~clean).sum(),singular[0]/singular[-1], abs(error@mode)/(np.linalg.norm(error)*np.linalg.norm(mode)))
        fits = [p, *(refit(p,x,axes,c) for c in (3.5,4.5,5.5)),oracle.x]
        for k,q in zip(outcomes,fits):
            m._fit = lambda x,axes: q
            outcomes[k].append(errors(m.calibrate_and_estimate_heading(s),g))
        m._fit = fit_current
    for key, values in outcomes.items():
        report(f'{lo}-{hi-1} {key}',values)
        print('bias failures:',(np.flatnonzero(np.asarray(values)[:,1]>.35)+lo).tolist())
