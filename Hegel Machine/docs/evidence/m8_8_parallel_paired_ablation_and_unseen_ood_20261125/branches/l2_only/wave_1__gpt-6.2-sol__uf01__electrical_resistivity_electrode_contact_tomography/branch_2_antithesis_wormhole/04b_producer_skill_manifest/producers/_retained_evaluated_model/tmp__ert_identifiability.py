from pathlib import Path
s = Path('/tmp/ert_stress.py').read_text()
prefix = s[s.index('rng ='):s.index('for key, rows')]
prefix = prefix.replace('        test(regime, dict(L_m=L,D_m=D,a0_m=a,z_tx_m=t,z_rx_m=r,V_obs=v),[x,z],rx)', '''        if regime in ('interior', 'changed geometry', 'low amplitude'):
            diagnose(regime, rep, v, spikes, L, D, a, t, r, x, z, amp)''')
import numpy as np
from scipy.linalg import helmert
from ert_solver import _Profile, invert_ert_anomaly
results = {}
def diagnose(regime, rep, v, spikes, L, D, a, t, r, x, z, amp):
    ii, jj = np.indices(v.shape)
    ii, jj = ii.ravel(), jj.ravel()
    tx, rx = t[ii], r[jj]
    d = rx-tx
    R = np.hypot(L,d)
    A = np.column_stack((L/R, (d/R)**2*L/R, helmert(len(t)).T[ii], helmert(len(r)).T[jj]))
    g = np.array([x,z])
    keep = np.ones(v.size,bool)
    keep[spikes] = False
    p = _Profile(A,v.ravel(),keep,tx,rx,L,D,a)
    h = p.search(a)*[L,D]
    w = np.array(invert_ert_anomaly(dict(L_m=L,D_m=D,a0_m=a,z_tx_m=t,z_rx_m=r,V_obs=v))['anomaly_xz_m'])
    k,aa,bb,tt,rr = p.kernel(x,z)
    J = np.column_stack([p.project(k), p.project(amp*k*((L-x)/bb-x/aa)), p.project(-amp*k*(tt/aa+rr/bb))])
    sx = .00023*np.sqrt(np.linalg.inv(J.T@J)[1,1]); sz = .00023*np.sqrt(np.linalg.inv(J.T@J)[2,2])
    def sep(alt):
        q=p.project(p.kernel(*alt)[0]); y=amp*p.project(k)
        scale = max(0.,q@y)/(q@q)
        return np.linalg.norm(y-q*scale)/.00023, scale
    results.setdefault(regime,[]).append((np.linalg.norm(w-g), np.linalg.norm(h-g), sx, sz, rep, g, h, sep(h)))
exec(prefix)
for key, rows in results.items():
    nums=np.array([r[:4] for r in rows]); worst=max(rows,key=lambda r:r[0]); print(key,'max blind/oracle and p90 Fisher x/z',nums[:,:2].max(0),np.quantile(nums[:,2:],.9,axis=0),'maximum blind-oracle error difference',np.max(abs(nums[:,0]-nums[:,1])),'worst',worst)
