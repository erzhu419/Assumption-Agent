import pickle
import numpy as np
from scipy.optimize import least_squares, nnls
import prony_rheology as p

x = pickle.load(open('public_rheology.pkl', 'rb'))
e = x['experiment']
t, s, y, u, v = (e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain'))
d = np.c_[np.ones_like(t), (t-t.mean())/np.ptp(t)]
grid = np.geomspace(.02, 20, 60)
_, r = p._calibrate(p._design(t,s,grid),d,y)
keep = np.abs(r-np.median(r)) < 4.5*p._noise(r)
print('inliers and noise', keep.sum(), p._noise(r))

# The same fixed inlier set isolates the effect of model complexity and time coverage.
def solve(mask, start, fixed_last=None):
    q = np.linalg.qr(d[mask], mode='reduced')[0]
    b = y[mask] - q @ (q.T @ y[mask])
    def residual(z, full=False):
        tau = np.exp(z)
        if fixed_last is not None:
            tau = np.r_[tau, fixed_last]
        a = p._design(t,s,tau)[mask]
        a -= q @ (q.T @ a)
        norms = np.maximum(np.linalg.norm(a,axis=0), 1e-12)
        g = nnls(a/norms,b)[0]/norms
        return (tau,g,b-a@g) if full else b-a@g
    opt = least_squares(residual,np.log(start),bounds=(np.log(.02),np.log(20)),max_nfev=65)
    return residual(opt.x,True)

def report(label, mask, tau, g, r):
    pred = p._design(u,v,tau)@g
    truth = x['gt_test_stress']
    train = p._design(t,s,tau)@g
    beta = np.linalg.lstsq(d[mask],(y-train)[mask],rcond=None)[0]
    valid = keep & ~mask
    cv = np.sqrt(np.mean((y-train-d@beta)[valid]**2)) if np.any(valid) else np.nan
    print(label, 'tau',np.round(tau,3),'g',np.round(g,3),'train_rms',round(np.sqrt(np.mean(r*r)),6),'heldout_rms',round(cv,6),'stress_rel',round(np.linalg.norm(pred-truth)/np.linalg.norm(truth),5))

for start in ([.2,8], [.2,1.4,9], [.04,.26,1.5,10], [.18,1.2,4,13]):
    report('free',keep,*solve(keep,start))
for slow in (5,6,7,8,9,10,11,13,16,20):
    report(f'fixed slow={slow}',keep,*solve(keep,[.20,1.4],slow))
for a,b in ((0,2),(2,6),(6,10),(10,14),(14,18),(18,22),(22,26),(26,31)):
    mask = keep & ~((t>=a)&(t<b))
    report(f'leave {a}-{b}',mask,*solve(mask,[.2,1.4,9]))
_,g,r = solve(keep,[.2,1.4,9]); print('inlier residual lag correlation',np.corrcoef(r[1:],r[:-1])[0,1])
