import pickle, time
import numpy as np
from ert_solver import invert_ert_anomaly as masked
from ert_coverage import invert_ert_anomaly as recovered

public = pickle.load(open('/holdout_task/public_ert_surveys.pkl','rb'))
results = {}
def test(label, s, truth, rx):
    for name, fn in [('mask', masked), ('coverage', recovered)]:
        t = time.perf_counter()
        ans = fn(s)
        elapsed = time.perf_counter()-t
        pos = np.linalg.norm(np.asarray(ans['anomaly_xz_m'])-truth)
        volts = np.sqrt(np.mean((np.asarray(ans['rx_contact_offset_v'])-rx)**2))
        results.setdefault((label, name), []).append((pos, volts*1e3, elapsed))

for item in public:
    s, g = item['survey'], item['gt']
    test('public', s, g['anomaly_xz_m'], g['rx_contact_offset_v'])

import sys
rng = np.random.default_rng(int(sys.argv[1]))
for regime in ('interior', 'near walls', 'variable spikes', 'changed geometry', 'low amplitude', 'strong anomaly'):
    for rep in range(48):
        L, D, a, m, n = 36., 72., 6., 14, 15
        amp = rng.uniform(6., 8.)
        if regime == 'changed geometry':
            L,D,a = rng.uniform(28,48),rng.uniform(60,95),rng.uniform(4,8)
            m,n = (int(q) for q in rng.integers(11,20,2))
        x,z = rng.uniform(.2,.8)*L,rng.uniform(.19,.81)*D
        if regime == 'near walls':
            x = rng.uniform(.025,.18)*L
            if rep%2: x = L-x
        if regime == 'low amplitude': amp = rng.uniform(3.,5.)
        if regime == 'strong anomaly': amp = rng.uniform(12.,22.)
        t=np.linspace(.083,.917,m)*D
        r=np.linspace(.069,.931,n)*D
        d=r[None,:]-t[:,None]
        R=np.hypot(L,d)
        tx,rx = rng.normal(0,.008,m),rng.normal(0,.008,n)
        tx -= tx.mean(); rx -= rx.mean()
        v = rng.uniform(1.5,2.4)/R + rng.uniform(-.5,.5)*d*d/R**3 + tx[:,None] + rx
        v += amp / (np.sqrt(x*x+(z-t[:,None])**2+a*a)*np.sqrt((L-x)**2+(z-r)**2+a*a))
        v += rng.normal(0,.00023,v.shape)
        spikes=rng.choice(m*n,int(.05*m*n),replace=False)
        step = .06 if regime != 'variable spikes' else rng.uniform(.05,.07,len(spikes))
        v.flat[spikes] += rng.choice([-1,1],len(spikes))*step
        test(regime, dict(L_m=L,D_m=D,a0_m=a,z_tx_m=t,z_rx_m=r,V_obs=v),[x,z],rx)
for key, rows in results.items():
    a=np.asarray(rows)
    print(key,'p90',np.round(np.quantile(a,.90,axis=0),5),'max',np.round(a.max(axis=0),5),'fail',np.sum(a[:,:2]>[2,1.2],axis=0),'rms',np.round(np.sqrt(np.mean(a[:,:2]**2,axis=0)),5),'worst rep',np.argmax(a[:,:2],axis=0))
