"""Carrier-normalized pitch-catch localization."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares
from scipy.signal import hilbert


def _analytic(y):
    return hilbert(np.asarray(y, float), axis=0)


def _samples(y, t, period):
    i = np.broadcast_to(np.arange(y.shape[1]), np.shape(t))
    return map_coordinates(y, [t * period, i], order=1, mode='constant')


def _arrival(z, d, period, prior=None):
    env = abs(z)
    t = np.arange(len(z)) / period
    valid = (t >= 1.4) & (t <= t[-1] - .8)
    peaks = t[valid][env[valid].argmax(axis=0)]
    def fit(peaks, start):
        low, high = (0., np.inf) if prior is None else (max(0., prior[0]-1.3), prior[0]+1.3)
        start = [np.clip(start[0], low+1e-7, high-1e-7), max(start[1], 1e-7)]
        def residual(q):
            e = q[0]+q[1]*d-peaks
            if prior is not None:
                e = np.r_[e, .025 * (q[1]-prior[1])]
            return e
        return least_squares(residual, start, bounds=([low,0.],[high,np.inf]), loss='soft_l1', f_scale=.13).x
    q = fit(peaks, np.polyfit(d, peaks, 1)[::-1] if np.ptp(d)>1e-9 else [2., 8.])
    for _ in range(2):
        guess = q[0]+q[1]*d
        offsets = np.linspace(-.8,.8,33)
        scan = guess + offsets[:,None]
        peaks = scan[_samples(env, scan, period).argmax(axis=0), np.arange(len(d))]
        q = fit(peaks,q)
    return q


def _geometry(s, p, plate):
    a, b = s[p[:,0]], s[p[:,1]]
    d = np.linalg.norm(a-b,axis=1)
    images = []
    for ax in range(2):
        for edge in (0.,plate[ax]):
            m = b.copy(); m[:,ax] = 2*edge-m[:,ax]
            images.append(np.linalg.norm(a-m, axis=1))
    return a,b,d,np.array(images)


def _pair_reduce(p, z):
    p = np.sort(p,axis=1)
    u, idx = np.unique(p,axis=0,return_inverse=True)
    return u, np.stack([np.mean(z[:,idx==i],axis=1) for i in range(len(u))],axis=1)


def _reference(b, scale, period, tr, p, z):
    r = b.get('reference')
    if r is None:
        return None, None, None
    rs = np.asarray(r['ref_transducers'],float)/scale
    rp, rz = _pair_reduce(np.asarray(r['ref_pairs']),_analytic(r['ref_traces']))
    plate = np.asarray(r['ref_plate'],float)/scale
    _,_,rd,_ = _geometry(rs,rp,plate)
    q = _arrival(rz,rd,period)
    h0,h1 = r.get('ref_thickness_mm'), b.get('target_thickness_mm')
    if h0 and h1:
        q[1] *= np.sqrt(h0/h1)
    off = np.arange(-2.,2.001,.05)
    raw_q = _arrival(rz,rd,period)
    pulses = _samples(abs(rz), raw_q[0] + raw_q[1]*rd+off[:,None],period)
    pulse = (off,np.median(pulses / np.maximum(pulses.max(axis=0),1e-12),axis=1))
    same = np.allclose(plate,np.asarray(b['target_plate'])/scale,rtol=0,atol=1e-8) and rs.shape==tr.shape and np.allclose(rs,tr,rtol=0,atol=1e-8)
    if same and h0 and h1:
        same = abs(h0/h1-1) < 1e-6
    if same and len(rz)==len(z):
        ix = {tuple(k):v for v,k in enumerate(rp)}
        if all(tuple(k) in ix for k in p):
            twin = rz[:,[ix[tuple(k)] for k in p]]
            n = min(len(z)-1, int(np.median(raw_q[0]+raw_q[1]*rd+2)*period))
            x,y = z[n:].real.ravel(),twin[n:].real.ravel()
            x,y = x-x.mean(),y-y.mean()
            if x@y > .78*np.linalg.norm(x)*np.linalg.norm(y):
                return q,pulse,twin
    return q,pulse,None


def _projector(p,d,n):
    sensors = np.eye(n)[p].sum(axis=1)
    std = np.maximum(sensors.std(axis=0),.05)
    sensors = (sensors-sensors.mean(axis=0))/std
    bg = np.column_stack((np.ones(len(d)),np.log(d),d,sensors))
    # Gain effects are regularized in standardized coordinates; geometric loss is shared.
    ridge = np.diag(np.r_[0.,.08,.08,1.8*np.ones(n)])*len(d)
    return np.eye(len(d))-bg@np.linalg.solve(bg.T@bg+ridge,bg.T)


def _prepare(b):
    scale = float(max(b['target_plate']))
    plate = np.asarray(b['target_plate'],float)/scale
    s = np.asarray(b['target_transducers'],float)/scale
    period = float(b['fs_hz']/b['f0_hz'])
    p,z = _pair_reduce(np.asarray(b['target_pairs']),_analytic(b['target_traces']))
    a,v,d,wall = _geometry(s,p,plate)
    prior,pulse,twin = _reference(b,scale,period,s,p,z)
    q = _arrival(z, d, period, prior)
    td = q[0]+q[1]*d
    tw = q[0]+q[1]*wall
    env = abs(z)
    offsets = np.linspace(-.65,.65,27)
    weights = np.exp(-.5*(offsets/.46)**2) if pulse is None else np.interp(offsets,*pulse)
    amp = np.sqrt(np.average(_samples(env,td+offsets[:,None],period)**2,axis=0,weights=weights))
    if twin is None:
        proj = _projector(p,d,len(s))
        shadow = -np.log(np.maximum(amp,1e-12))
        x = env/np.maximum(amp,1e-12)
        x = gaussian_filter1d(x,.25*period,axis=0)
        # Remove the direct packet and reduce evidence near predicted wall returns.
        t = np.arange(len(z))[:,None]/period
        gate = 1-np.exp(-.5*((t-td)/1.35)**6)
        gate *= np.where(t>td,1.,0.)
        for w in tw:
            gate *= 1-.87*np.exp(-.5*((t-w)/.65)**2)
        x = x*gate
    else:
        proj = np.eye(len(p))
        ra = np.sqrt(np.average(_samples(abs(twin),td+offsets[:,None],period)**2,axis=0,weights=weights))
        shadow = np.log(np.maximum(ra,1e-12)/np.maximum(amp,1e-12)); shadow -= np.median(shadow)
        x = gaussian_filter1d(abs(z-twin),.22*period,axis=0)
        t = np.arange(len(z))[:,None]/period
        x *= 1-np.exp(-.5*((t-td)/.6)**2)
    return scale,plate,s,p,a,v,d,q,period,proj,shadow,x,twin is not None


def _fields(pre, points):
    _,_,s,p,a,b,d,q,period,proj,shadow,x,twin = pre
    delta = points[:,None,:]-a
    u = np.clip(np.sum(delta*(b-a),axis=2)/d**2,0,1)
    r = np.linalg.norm(delta-u[...,None]*(b-a),axis=2)
    num = proj@shadow
    length = np.linalg.norm(delta,axis=2)+np.linalg.norm(points[:,None,:]-b,axis=2)
    w = np.array([.025,.05,.085]) * 1.0
    sh = 0
    for width in w:
        k = np.exp(-.5*(r/width)**2)
        norm = np.sqrt(np.maximum(np.sum((k@proj)*k,axis=1),1e-12))
        sc = (k*num).sum(axis=1)/norm
        sh = np.maximum(sh,sc)
    td = q[0]+q[1]*d
    ts = q[0]+q[1]*length
    xs = np.mean([_samples(x,ts+o,period) for o in (-.35,0,.35)],axis=0)
    # Fit wave amplitudes independently of phase, but hold their travel times in common.
    cd = np.mean(xs,axis=1)
    return sh,cd


def _scatter_map(b,pre,points,wall_ridge=.025):
    L,plate,s,p,a,v,d,q,F,P,y,x,twin=pre
    _,z=_pair_reduce(b['target_pairs'],_analytic(b['target_traces']))
    t=np.arange(len(z))/F
    base=z*np.exp(-2j*np.pi*t[:,None])
    td=q[0]+q[1]*d
    tw=q[0]+q[1]*_geometry(s,p,plate)[-1]
    offsets=np.linspace(-2.6,2.6,105)
    def profile(base,td):
        samples=_samples(base,td+offsets[:,None],F)
        mid=samples[len(offsets)//2]
        normalized=samples/np.where(abs(mid)>1e-12,mid,1e-12)
        return np.median(normalized.real,axis=1)+1j*np.median(normalized.imag,axis=1)
    pulse=profile(base, td)
    # Gaussian taper keeps the learned pulse from incorporating coherent late echoes.
    fit=(abs(offsets)<1.5)
    phase=np.polyfit(offsets[fit],np.unwrap(np.angle(pulse))[fit],2)
    curvature = np.polyfit(offsets[fit]**2, np.log(np.maximum(abs(pulse[fit]),.03)), 1)[0]
    sigma = np.sqrt(-.5 / np.clip(curvature, -1.4, -.19))
    def atom(u):
        return np.exp(-.5*(u/sigma)**2+1j*(phase[0]*u*u+phase[1]*u))
    U=t[None,None,:]-np.vstack([td,tw]).T[:,:,None]
    A=atom(U)
    deriv=U*atom(U)/sigma
    A=np.concatenate([A,deriv],axis=1)
    gram=A.conj()@A.transpose(0,2,1)
    diag=gram.diagonal(axis1=1,axis2=2).real.copy()
    diag=np.maximum(diag,.15*F)
    ridge=np.array([.002]+[wall_ridge]*4+[.075]+[.32]*4)
    inv=np.linalg.inv(gram+np.eye(10)[None]*diag[:,None]*ridge)
    coef=(inv@(A.conj()@base.T[:,:,None]))[:,:,0]
    R=base.T-(coef[:,None,:]@A)[:,0,:]
    times=np.arange(0,t[-1]+.001,.1)
    bank=atom(t[None,:]-times[:,None]); n=(abs(bank)**2).sum(axis=1)
    dot=R@bank.conj().T
    proj=A.conj()@bank.T
    leverage=np.einsum('pij,pij->pj',proj.conj(),inv@proj).real
    norm=np.maximum(n-leverage,.06*F)
    amp=np.maximum(abs(coef[:,0]),.012)
    # Match shifted packets with free complex amplitudes; suppress unresolved modes.
    improvement=abs(dot)**2/norm
    detect= np.sqrt(improvement/np.maximum(n,.1))*np.clip(norm/np.maximum(n,.1),0,1)**.25
    ts=q[0]+q[1]*(np.linalg.norm(points[:,None]-a,axis=2)+np.linalg.norm(points[:,None]-v,axis=2))
    def field(x):
        ar=ts/.1; j=np.clip(ar.astype(int),0,len(times)-2); frac=np.clip(ar-j,0,1); k=np.arange(len(p));
        return x[k,j]*(1-frac)+x[k,j+1]*frac
    valid = ts <= t[-1] - .6
    depth = detect / np.maximum(amp[:, None], 1e-12)
    coda = np.mean(field(detect) * valid, axis=1)
    return coda, np.quantile(depth, .99)

def _standardize(x):
    median = np.median(x)
    return (x - median) / max(np.quantile(x, .98) - median, 1e-12)


def _shadow_map(pre, points):
    _, _, s, p, a, b, d, _, _, proj, y, _, _ = pre
    v = b - a
    offset = points[:, None] - a
    u = np.clip(np.einsum('gpi,pi->gp', offset, v) / d**2, 0, 1)
    r2 = np.sum((offset - u[..., None] * v)**2, axis=2)
    g = np.eye(len(s))[p].sum(axis=1)
    g = (g - g.mean(axis=0)) / np.maximum(g.std(axis=0), .05)
    f = np.column_stack((np.log(d), d))
    f = (f - f.mean(axis=0)) / np.maximum(f.std(axis=0), 1e-12)
    design = np.column_stack((np.ones(len(d)), f, g))
    penalties = len(d) * np.diag(np.r_[0., .01, .01, np.full(len(s), 1.8)])
    distance_fit = np.linalg.solve(design.T @ design + penalties, design.T)
    maps = []
    for width in (.017, .025, .036):
        k = np.exp(-.5 * r2 / width**2)
        for bg in (proj, np.eye(len(d)) - design @ distance_fit):
            kp = np.einsum('gp,pq->gq', k, bg, optimize=False)
            score = np.einsum('gp,p->g', kp, y) / np.sqrt(np.maximum(np.sum(kp*k, axis=1), 1e-12))
            maps.append(_standardize(score))
    return np.max(maps, axis=0)


def _crossing_shadow(bundle, pre, points, shadow):
    """Combine ridge-adjusted attenuation with support from nonparallel rays."""
    _, plate, sensors, pairs, a, b, distance, timing, period, proj, loss, _, _ = pre
    _, traces = _pair_reduce(bundle['target_pairs'], _analytic(bundle['target_traces']))
    offsets = np.array([-2., -1.65, -1.3, -.95])
    direct = timing[0] + timing[1] * distance
    lead = _samples(abs(traces), direct + offsets[:, None], period)
    early_loss = -np.log(np.maximum(lead, 1e-12)).mean(axis=0)

    wall = _geometry(sensors, pairs, plate)[-1]
    delay = np.sort(timing[1] * (wall - distance), axis=0)[:2]
    overlap = np.exp(-.5 * (delay / 1.15)**2)
    phase = 2 * np.pi * delay
    echo = np.column_stack((overlap.T, (overlap*np.cos(phase)).T,
                           (overlap*np.sin(phase)).T))
    echo -= echo.mean(axis=0)
    pe = proj @ echo
    fit = np.linalg.solve(echo.T @ pe + .2*len(distance)*np.eye(echo.shape[1]), pe.T)
    residual = (proj - pe @ fit) @ (.5 * (loss + early_loss))
    weight = 1 - .88 * np.exp(-.5 * (delay[0] / 1.05)**2)

    ray = b - a
    direction = ray / distance[:, None]
    ux, uy = direction.T
    crossing = (ux[:, None]*uy - uy[:, None]*ux)**2
    offset = points[:, None] - a
    u = np.clip(np.einsum('gpi,pi->gp', offset, ray) / distance**2, 0, 1)
    dist_sq = np.sum((offset - u[..., None]*ray)**2, axis=-1)
    kernel = np.exp(-.5 * dist_sq / .024**2)
    positive = kernel * weight * np.maximum(residual, 0)
    agreement = np.einsum('gp,pq,gq->g', positive, crossing, positive, optimize=False)
    support = np.sqrt(np.maximum(agreement, 0)) / (1 + .55*(kernel*weight).sum(axis=1))**.8
    return .6 * _standardize(support) + .4 * shadow


def estimate_flaw_xy(bundle: dict) -> tuple[float, float]:
    """Estimate flaw position in the length unit of ``target_plate``."""
    pre = _prepare(bundle)
    scale, plate = pre[:2]
    axes = [np.linspace(0, w, max(2, round(w/.009) + 1)) for w in plate]
    points = np.array(np.meshgrid(*axes)).reshape(2, -1).T
    if pre[-1]:
        shadow, coda = _fields(pre, points)
        score = .55 * _standardize(shadow) + _standardize(coda)
    else:
        shadow = _crossing_shadow(bundle, pre, points, _shadow_map(pre, points))
        coda, depth = _scatter_map(bundle, pre, points)
        scatter_weight = .08 + 3.8 * depth**4 / (.080**4 + depth**4)
        score = shadow + scatter_weight * _standardize(coda)
    return tuple(map(float, points[np.argmax(score)] * scale))
