import numpy as np
from scipy.signal import correlate, find_peaks
from scipy.optimize import least_squares

def reconstruct_two_absorbers_mm(scan: dict) -> dict:
    y=np.asarray(scan['signals'],float); sensors=np.asarray(scan['sensor_xy_mm'],float)
    pin=np.asarray(scan['calib_pin_xy_mm'],float); fs=float(scan['fs_hz'])
    n,nch=y.shape
    u=np.arange(-12.,13.); w=-u*np.exp(-.5*(u/3.2)**2); w-=w.mean(); w/=np.linalg.norm(w)
    mf=np.array([correlate(y[:,j]-y[:,j].mean(),w,mode='same',method='fft') for j in range(nch)])
    strength=np.abs(mf); dp=np.linalg.norm(sensors-pin,axis=1)
    # Calibrate from the pin's direct path across the sensor ring.
    best=(-1,None,None)
    for c in np.arange(1435.,1545.01,.05):
        pred=dp*fs*1e-6/(c/1000.)
        for off in np.arange(-5.,6.,.25):
            ix=np.rint(pred+off).astype(int); idx=np.clip(ix[:,None]+np.arange(-3,4),0,n-1)
            amp=strength[np.arange(nch)[:,None],idx].max(axis=1)
            score=np.sort(amp)[-12:].sum()
            if score>best[0]: best=(score,c,off)
    c=best[1]/1000.; t0=best[2]
    arr=[]
    for j in range(nch):
        pk,_=find_peaks(strength[j],distance=12,prominence=.18)
        arr.append(pk[(pk>80)&(pk<n-30)].astype(float))
    def fit_sources(seed):
        def fun(z):
            out=[]
            for p in z.reshape(2,2):
                pred=t0+np.linalg.norm(sensors-p,axis=1)*fs*1e-6/c
                for j in range(nch):
                    if len(arr[j]):
                        d=arr[j]-pred[j]; out.append(d[np.argmin(np.abs(d))])
            return np.asarray(out)
        return least_squares(fun,seed,bounds=([-31]*4,[31]*4),loss='soft_l1',f_scale=1.5,max_nfev=160)
    rng=np.random.default_rng(19); winner=None
    for _ in range(50):
        f=fit_sources(rng.uniform(-24,24,4)); pts=f.x.reshape(2,2)
        score=np.median(np.abs(f.fun))+max(0.,5.-np.linalg.norm(pts[0]-pts[1]))
        if winner is None or score<winner[0]: winner=(score,pts)
    pts=sorted([[float(p[0]),float(p[1])] for p in winner[1]],key=lambda q:q[0])
    return {'sound_speed_m_per_s':float(c*1000.),'absorbers_xy_mm':pts}
