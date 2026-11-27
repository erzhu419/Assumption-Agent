exec(open('/tmp/chbasis.py').read().split('C,E=profile(tau)')[0])
lab = di['lab_id']
raw = di['coupon_features_raw']
xv = raw.copy()
for l in np.unique(lab):
    mask = lab == l
    refs = di['ref_coupons'][mask]
    xv[mask] = (xv[mask] - np.median(refs.mean(0))) / np.median(refs.std(0))
xv = np.where(np.isfinite(xv), xv, np.nanmean(xv, axis=0))
P = basis(tau)
Q = Y - hd[:, None] * (P[:, :2] @ beta[:2])
r = beta[2] + xv @ beta[3:]
_, rate_design = m._design(tau, xv, pd, hd, dt, t)
z = hd[:, None] - rate_design[:, :2] * tau

def features(kind, a=0.):
    F = np.column_stack((np.ones(len(r)), r))
    extras = {'affine': [], 'pairwise': [z[:,0]-z[:,1], hd-z[:,0]], 'cross': [r*(z[:,0]-z[:,1]),r*hd], 'three-way': [r*hd*(z[:,0]-z[:,1])], 'curvature': [r*r,r**3], 'lab-centered': [r-np.array([np.mean(r[lab==l]) for l in lab])], 'rational': [r/(1+a*r*r)-r]}
    return np.column_stack([F,*extras[kind]])

def predict(F, fit, val):
    B = np.linalg.lstsq(F[fit], Q[fit], rcond=None)[0] @ np.linalg.pinv(P.T)
    return F[val] @ B @ P.T

for name in ('affine','pairwise','cross','three-way','curvature','lab-centered','rational'):
    e = np.empty_like(Q)
    pars = []
    for l in np.unique(lab):
        fit = lab != l
        if name == 'rational':
            def inner(a):
                F = features(name,a)
                loss = []
                for j in np.unique(lab[fit]):
                    sk = lab == j
                    err = Q[sk]-predict(F,fit & ~sk,sk)
                    loss.append(np.mean(err[:,:-1]**2)+np.mean(err[:,-1]**2))
                return np.mean(loss)
            a = min((.05,.2,1.,4.), key=inner)
            pars.append(a)
        else:
            a = 0.
        e[~fit] = Q[~fit]-predict(features(name,a),fit,~fit)
    print(name,'RMSE and max curve/rate',np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.max(abs(e[:,:-1])),np.max(abs(e[:,-1]))],6),'inner rational settings',pars)
    if name == 'affine':
        for l in np.unique(lab):
            k = lab == l
            print('  lab',l,'signed mean curve/rate',np.round([e[k,:-1].mean(),e[k,-1].mean()],6))
        for label,k in [('negative drift',r<0),('positive drift',r>=0)]:
            if k.any(): print(' ',label,int(k.sum()),'max curve/rate',np.round([np.max(abs(e[k,:-1])),np.max(abs(e[k,-1]))],6))

cal = d['calibration_load_cells']
xcal = cal['bridge_raw']; ycal = cal['true_channel_loads'].sum(1)
F = np.column_stack((np.ones(len(xcal)), xcal))
G = np.column_stack([F]+[xcal[:,j]*xcal[:,k] for j in range(4) for k in range(j,4)])
groups = np.argsort(np.argsort(ycal)) % 5
for label,design in [('affine bridge',F),('quadratic bridge',G)]:
    e = np.empty_like(ycal)
    for j in range(5):
        k = groups == j
        e[k] = ycal[k]-design[k] @ np.linalg.lstsq(design[~k],ycal[~k],rcond=None)[0]
    print(label,'total stress CV RMS/max', np.round([np.sqrt(np.mean(e**2)),np.max(abs(e))],7))
