import pickle, time
import numpy as np
from scipy.linalg import helmert
from ert_solver import invert_ert_anomaly

items=pickle.load(open('/holdout_task/public_ert_surveys.pkl','rb'))
rng=np.random.default_rng(927)
def assess(s,p,r):
    t=time.monotonic(); a=invert_ert_anomaly(s)
    return [np.linalg.norm(np.asarray(a['anomaly_xz_m'])-p), np.sqrt(np.mean((a['rx_contact_offset_v']-r)**2)), time.monotonic()-t]
for k,d in enumerate(items):
    s,g=d['survey'],d['gt']; L,D,a=s['L_m'],s['D_m'],s['a0_m']
    tt,rr=s['z_tx_m'],s['z_rx_m']; y=s['V_obs']; m,n=y.shape
    p=np.array(g['anomaly_xz_m']); r=np.array(g['rx_contact_offset_v'])
    scores=[]
    for axis,N in enumerate((m,n)):
        for i in range(N):
            t=s.copy(); mask=np.arange(N)!=i
            if axis==0:
                t.update(z_tx_m=tt[mask], V_obs=y[mask,:]); ref=r
            else:
                t.update(z_rx_m=rr[mask], V_obs=y[:,mask]); ref=r[mask]-r[mask].mean()
            scores.append(assess(t,p,ref))
    scores=np.array(scores)
    print('public',k,'electrode jackknife max pos, rx, sec',scores.max(0), 'position worst axis/index',[(0,i) for i in range(m)]+[(1,i) for i in range(n)] if False else np.argmax(scores[:,0]))
    dz=rr[None,:]-tt[:,None]; R=np.hypot(L,dz)
    A=np.column_stack((1/R.ravel(), (dz**2/R**3).ravel(), np.repeat(helmert(m).T,n,axis=0)))
    def kernel(p):
        x,z=p
        return 1/np.sqrt((x*x+(z-tt[:,None])**2+a*a)*((L-x)**2+(z-rr[None,:])**2+a*a))
    B=np.column_stack((A,kernel(p).ravel())); target=(y-r).ravel(); use=np.ones(m*n,bool)
    for _ in range(3):
        c=np.linalg.lstsq(B[use],target[use],rcond=None)[0]; res=target-B@c; use=np.abs(res)<.03
    sig=np.std(res[use]); print('public',k,'c0/c1/amp',c[[0,1,-1]],'clean sigma',sig,'spikes',np.sum(~use),'spike magnitudes',np.quantile(np.abs(res[~use]),[0,.5,1]))
    base=(A@c[:-1]).reshape(m,n)+r
    for region, positions in [('interior',[(9,16),(18,36),(26,55),(12,52),(25,20)]),('edges',[(0,36),(2,10),(34,62),(36,36),(18,0),(18,72)])]:
        rows=[]
        for pos in positions:
            for rep in range(6):
                v=base+c[-1]*kernel(pos)+rng.normal(0,sig,(m,n)); idx=rng.choice(v.size,round(.05*v.size),replace=False); v.flat[idx]+=rng.choice([-.06,.06],len(idx))
                rows.append(assess(dict(s,V_obs=v),pos,r))
        rows=np.array(rows); print(k,region,'max',rows.max(0),'p90',np.quantile(rows,.9,axis=0),'p above target',np.sum(rows[:,0]>2),'worst position',positions[np.argmax(rows[:,0])//6])
