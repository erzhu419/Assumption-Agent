import pickle
import numpy as np
from scipy.optimize import least_squares
from prony_probe import basis, solve

p = pickle.load(open('/holdout_task/public_rheology.pkl', 'rb'))
x = p['experiment']
t, e, y = (x[k] for k in ('train_t', 'train_strain', 'train_stress_raw'))
tt, et = (x[k] for k in ('test_t', 'test_strain'))
d = np.column_stack((np.ones(t.size), (t-t.mean())/np.ptp(t)))
grid = np.geomspace(.02, 20, 80)
a = basis(t, e, grid, 'ramp')
lo = np.r_[np.zeros(4), [-np.inf]*2, [np.log(.02)]*3]
hi = np.r_[[np.inf]*6, [np.log(20)]*3]

def stress(v, T=t, E=e):
    return basis(T, E, np.exp(v[6:]), 'ramp') @ v[:4]

def fit(Y, good, start, scale, strength):
    # Penalize only long extrapolation relative to the actual calibration duration.
    def res(v):
        z = (stress(v) + d @ v[4:6] - Y)[good] / scale
        if strength:
            z = np.r_[z, np.sqrt(strength) * np.exp(v[6:]) / (np.ptp(t)/3)]
        return z
    return least_squares(res, start, bounds=(lo, hi), max_nfev=90, ftol=1e-8).x

strengths = (0, .5, 1, 2, 4)
starts = []
for lo_t, hi_t in [(0, 0), (0, 2), (2, 6), (6, 12), (12, 18), (18, 24), (24, 31)]:
    block = (t>=lo_t)&(t<hi_t)
    c,b,_,s = solve(a[~block],d[~block],y[~block])
    rr = y-a@c-d@b
    good = (~block)&(abs(rr-np.median(rr[~block]))<6*s)
    start = np.r_[max(c[0], .01), 25., 15., 10., b, np.log([.2, 1.5, 10.])]
    v = fit(y, good, start, s, 0)
    if lo_t == hi_t:
        full = v
        full_good = good
        scale = s
    else:
        starts.append((block,good,start,s))
for l in strengths:
    v = fit(y,full_good,full,scale,l)
    cv=[]
    for block,good,start,s in starts:
        w=fit(y,good,start,s,l)
        rr=(stress(w)+d@w[4:6]-y)[block]
        rr=rr[abs(rr)<6*s]
        cv.append(np.sqrt(np.mean(rr**2)))
    z=p['gt_test_stress']
    print('public',l,'G',round(v[0],4),'tau',np.exp(v[6:]).round(3),'testRRMSE',round(np.linalg.norm(stress(v,tt,et)-z)/np.linalg.norm(z),6),'blocked/worst',np.round(cv,5),round(max(cv),5))

# Synthetic recovery uses multiple spectrum shapes on the measured loading protocol.
# Use identical noise draws across penalties. Masks isolate the regularizer here;
# the next end-to-end candidate must recover the spike masks itself.
rng = np.random.default_rng(28471)
for G, times, amplitudes in [(7.,[.05,.9,4.],[19.,9.,5.]), (12.,[.22,1.5,7.],[25.,16.,10.]), (9.,[.14,1.7,10.],[30.,17.,12.]), (15.,[.29,2.9,15.],[30.,13.,11.]), (11.,[.11,3.,18.],[15.,11.,17.])]:
    truth = np.r_[G,amplitudes,[.2,4.2],np.log(times)]
    mu=stress(truth)+d@truth[4:6]
    z=stress(truth,tt,et)
    result={l:[] for l in strengths}
    for j in range(30):
        yy=mu+rng.normal(0,scale,t.size)
        good=np.ones(t.size,dtype=bool);good[rng.choice(t.size,12,replace=False)]=False
        w=fit(yy,good,truth,scale,0)
        for l in strengths:
            v=w if not l else fit(yy,good,w,scale,l)
            result[l].append(((v[0]-G)/G,np.linalg.norm(stress(v,tt,et)-z)/np.linalg.norm(z)))
    for l, errors in result.items():
        a1,a2=np.abs(errors).T
        print('synthetic',times,l,'G bias',round(np.mean(np.asarray(errors)[:,0]),4),'G median/p90',np.quantile(a1,[.5,.9]).round(4),'stress p90',round(np.quantile(a2,.9),4))
