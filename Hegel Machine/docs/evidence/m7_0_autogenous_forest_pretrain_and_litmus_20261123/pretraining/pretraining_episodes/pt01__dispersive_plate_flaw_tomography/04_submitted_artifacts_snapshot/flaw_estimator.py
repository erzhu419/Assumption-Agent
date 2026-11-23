"""Ultrasonic flaw localization by direct-ray attenuation and scattered arrivals."""
import numpy as np
from scipy.signal import hilbert, fftconvolve
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist


def _paths(s, p, plate):
    a, z = s[p].transpose(1, 0, 2)
    ds = [np.linalg.norm(a-z, axis=1)]
    for j in range(2):
        for w in (0, plate[j]):
            q = z.copy(); q[:, j] = 2*w-q[:, j]
            ds.append(np.linalg.norm(a-q, axis=1))
    return np.array(ds).T


def _pulse(u, width=13.5, chirp=-.0015):
    return np.exp(-.5*(u/width)**2 + 1j*chirp*u*u)


def _clock(Y, d):
    peaks = np.abs(Y).argmax(axis=0)
    k = np.polyfit(d, peaks, 1)[::-1]
    for _ in range(3):
        k = least_squares(lambda q:q[0]+q[1]*d-peaks,k,loss='soft_l1',f_scale=1.3).x
        peaks = (np.abs(Y)* (abs(np.arange(len(Y))[:,None]-k[0]-k[1]*d)<12)).argmax(axis=0)
    return k


def _fit(Y, D, clock, width=13.5, chirp=-.0015):
    times = clock[0]+clock[1]*D
    u = np.arange(len(Y))[None, :, None]-times[:, None, :]
    # Broaden successive packets with propagation time, rather than treating
    # the direct and all wall paths as copies of one pulse.
    transit = (times-clock[0]) / (width/13.5)
    w = np.clip(10.9+.041*transit, 11.2, 23.)*(width/13.5)
    k = np.clip(-.00060-.000011*transit, -.0020, -.00062) / (width/13.5)**2
    B = _pulse(u,w[:,None,:],k[:,None,:])
    H = B.conj().transpose(0,2,1)
    G = H@B
    A = np.linalg.solve(G+(.20*np.eye(D.shape[1])), (H@Y.T[:,:,None]))[:,:,0]
    return A, (Y.T-(B@A[:,:,None])[:,:,0]).T, G


def _prepare(b):
    s = np.asarray(b['target_transducers_cm']); p = np.asarray(b['target_pairs']); W = np.asarray(b['target_plate_cm']); y = np.asarray(b['target_traces'])
    scale = b['fs_hz']/1e6
    t = np.arange(len(y))/scale
    carrier = np.exp(-2j*np.pi*b['f0_hz']/1e6*t)[:,None]
    Y = hilbert(y,axis=0)*carrier
    D = _paths(s,p,W); d = D[:,0]
    clock = _clock(Y,d)/scale
    A,R,G = _fit(Y,D,clock*scale,13.5*scale,-.0015/scale**2)
    twin = False; ref = b.get('reference')
    if ref is not None and np.allclose(W,ref['ref_plate_cm']) and abs(b['target_thickness_mm']-ref['ref_thickness_mm']) < .001:
        ss = np.asarray(ref['ref_transducers_cm']); rp = np.asarray(ref['ref_pairs']); ry = np.asarray(ref['ref_traces'])
        if s.shape == ss.shape and np.allclose(s,ss) and len(ry)==len(y):
            lookup = {tuple(q):i for i,q in enumerate(rp)}
            if all(tuple(q) in lookup for q in p):
                rr = ry[:,[lookup[tuple(q)] for q in p]]
                gate = t[:,None]>clock[0]+clock[1]*d+25
                cov = np.sum(y*rr*gate,axis=0)
                corr = cov / np.sqrt(np.sum(y*y*gate,axis=0)*np.sum(rr*rr*gate,axis=0)+1e-24)
                twin = np.median(corr)>.80
                if twin:
                    R = hilbert(y-rr,axis=0)*carrier
                    Ar,_,_ = _fit(hilbert(rr,axis=0)*carrier,D,clock*scale,13.5*scale,-.0015/scale**2)
    off = np.arange(-int(50*scale),int(50*scale)+1)/scale
    wave = _pulse(off)
    C = fftconvolve(R,wave[::-1,None].conj(),mode='same',axes=0)/np.sum(abs(wave)**2)
    direct = abs(A[:,0]); typical = np.median(direct*np.sqrt(d))/np.sqrt(d)
    if not twin:
        # Remove per-pair carrier phase using the fitted direct packet.
        C *= np.conj(A[:,0])[None,:]/np.maximum(direct,1e-8)
    C = C/np.maximum(typical,1e-8)
    g = (p[:,:,None]==np.arange(len(s))).sum(axis=1).astype(float)
    a,z = s[p].transpose(1,0,2); v = z-a
    dx = s[None,:,:]-a[:,None,:]; u = np.sum(dx*v[:,None,:],axis=-1)/d[:,None]**2
    sh = ((u>.01)&(u<.99)&(np.linalg.norm(dx-u[:,:,None]*v[:,None,:],axis=-1)<.15)).sum(axis=1)
    logamp = np.log(np.maximum(direct,1e-8)*np.sqrt(d))
    if twin:
        logamp -= np.log(np.maximum(abs(Ar[:,0]),1e-8)*np.sqrt(d)); F=np.ones((len(p),1))
    else:
        F = np.column_stack((g,sh,d/30))
    # Profile out gain and geometrical spreading, from both data and predicted shadow.
    penalty = np.full(F.shape[1],.12); penalty[-1]=.01
    if not twin: penalty[-2]=.03
    P = np.eye(len(p))-F@np.linalg.solve(F.T@F+np.diag(penalty),F.T)
    return s,p,W,D,clock,scale,C,logamp,P,twin


def _maps(x, q, width=1.55, coherence=.65, phase_ratio=1.70):
    s,p,W,D,clock,scale,C,logamp,P,twin=q
    a,z=s[p].transpose(1,0,2); v=z-a; d=D[:,0]
    delta = x[:,None,:]-a
    u = np.clip(np.sum(delta*v,axis=2)/d**2,0,1)
    ds = np.sum((delta-u[:,:,None]*v)**2,axis=2)
    K = np.exp(-.5*ds/width**2)
    nom = -np.einsum("ij,j->i", K, P@logamp); den = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", K,P,K,optimize=False),1e-9))
    shadow = np.maximum(0,nom)/den
    dist = cdist(x,s); L = dist[:,p[:,0]]+dist[:,p[:,1]]
    sample = (clock[0]+clock[1]*L)*scale
    ij = np.clip(sample.astype(int),0,len(C)-2); col=np.arange(len(p)); r=np.clip(sample-ij,0,1)
    c = (1-r)*C[ij,col]+r*C[ij+1,col]
    if not twin:
        gap = (L-d)*clock[1]; wallgap = np.min(abs(L[:,:,None]-D[None,:,1:]),axis=2)*clock[1]
        weight = (1-np.exp(-.5*(gap/13)**2))*(1-.90*np.exp(-.5*(wallgap/10)**2))
    else:
        weight = np.ones_like(L)
    weight*= (sample<len(C)-3)
    amp = abs(c)
    clipped = np.minimum(amp,.22)
    coverage = np.maximum(weight.sum(axis=1),.30*len(p))
    scatter = np.sum(weight*clipped,axis=1)/coverage
    if not twin:
        # Extra path length predicts the relative phase up to one unknown
        # scatter phase; maximize over that common phase by taking a magnitude.
        phasor = np.exp(1j*phase_ratio*2*np.pi*.08*clock[1]*(L-d))
        aligned = np.sum(weight*c*phasor*(clipped/np.maximum(amp,1e-12)),axis=1)
        scatter = (1-coherence)*scatter+coherence*abs(aligned)/coverage
    return shadow,scatter


def _search(q, mix=None, coherence=.65, phase_ratio=1.70):
    W=q[2]; x,y=np.meshgrid(np.arange(.25,W[0],.5),np.arange(.25,W[1],.5)); X=np.column_stack((x.ravel(),y.ravel()))
    if mix is None: mix = (.1,1.) if q[-1] else (.30,1.)
    sh,sc = [np.concatenate(v) for v in zip(*(_maps(xx,q,coherence=coherence,phase_ratio=phase_ratio) for xx in np.array_split(X,max(1,len(X)//250))))]
    score = mix[0]*sh/np.sqrt(len(q[1]))+mix[1]*sc
    start = X[score.argmax()]
    def objective(x):
        a,b=_maps(x[None,:],q,coherence=coherence,phase_ratio=phase_ratio); return -float(a[0]*mix[0]/np.sqrt(len(q[1]))+b[0]*mix[1])
    opt=minimize(objective,start,method='Powell',bounds=[(max(.01,z-.8),min(w-.01,z+.8)) for z,w in zip(start,W)],options={'xtol':.04})
    return tuple(map(float,opt.x))


def estimate_flaw_xy_cm(bundle: dict) -> tuple[float, float]:
    """Return the estimated flaw position in centimetres."""
    return _search(_prepare(bundle))
