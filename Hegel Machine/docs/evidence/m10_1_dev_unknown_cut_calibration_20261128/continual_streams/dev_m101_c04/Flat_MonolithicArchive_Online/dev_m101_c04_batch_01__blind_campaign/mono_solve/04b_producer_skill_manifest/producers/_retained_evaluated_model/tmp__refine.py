import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
m = runpy.run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_00_solver.py')
t = s['train']; y, T = t['y'], t['traj']; cal = m['_calibration'](s)
z = m['_coordinates'](t['x'], cal); c = m['_cross'](z); D = np.c_[z,c]; X = np.c_[np.ones(len(z)),D]
ls = lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
rng = np.random.default_rng(44039); n = len(y)
splits = [(np.arange(56), np.arange(56,n))]
for _ in range(12):
    j = rng.permutation(n); splits.append((j[:56],j[56:]))
for h in [np.linalg.norm(z,axis=1), *z.T, *(-z).T, abs(c)]:
    j = np.argsort(h); splits.append((j[:56],j[56:]))
for name in ['sqrt_shared','sqrt_shared_shift','sqrt_through','tanh_shared','tanh_through','power_shared','power_through']:
    all_err = []; fit = []
    for a,b in splits:
        def design(p,i):
            h = D[i] @ p[:5]
            if name.endswith('shift') or name.endswith('through'): h = h+p[5]
            if name.startswith('tanh'): f = np.tanh(h)
            elif name.startswith('power'):
                power = p[-1]; f = h/(1+abs(h)**power)**(1/power)
            else: f = h/np.sqrt(1+h*h)
            if name.endswith('through'): return np.c_[X[i],f]
            return np.c_[D[i] @ p[:5], f]
        def residual(p):
            M = design(p,a); return M@ls(M,y[a])-y[a]
        seed = ls(D[a],y[a]); k = 5+int(name.endswith(('shift','through')))
        lo, hi = np.full(k,-8.),np.full(k,8.)
        if name.startswith('power'): lo, hi = np.r_[lo,1.2], np.r_[hi,4.]
        cand = []
        for gain in [.65,1.,1.5]:
            p = np.r_[gain*seed,np.zeros(k-5)]
            if name.startswith('power'): p = np.r_[p,2.]
            cand.append(least_squares(residual,p,bounds=(lo,hi),max_nfev=100))
        r = min(cand,key=lambda f:f.cost); w = ls(design(r.x,a),y[a]); e = abs(design(r.x,b)@w-y[b]); all_err.append(e.mean());fit.append((r.x,w,np.sqrt(np.mean(r.fun**2))))
    print(name,'canonical / random worst / stress worst',*[round(i,6) for i in (all_err[0],max(all_err[1:13]),max(all_err[13:]))], 'fit',tuple(np.round(i,5) for i in fit[0]), flush=True)
for intercept in (False,True):
    A = X[:,:5] if intercept else z
    for rank in [1,2,3,5]:
        err=[]; end=[]; horizon=[]
        for a,b in splits:
            Q,R=np.linalg.qr(A[a]); u,sv,v=np.linalg.svd(Q.T@T[a],full_matrices=False)
            pred=A[b]@np.linalg.solve(R,(u[:,:rank]*sv[:rank])@v[:rank]); e=abs(pred-T[b]); err.append(e.mean());end.append(e[:,-1].mean());horizon.append(e.mean(axis=0).max())
        print('traj',intercept,rank,'MAE canonical / random worst / stress worst / worst final / worst horizon',*[round(i,6) for i in (err[0],max(err[1:13]),max(err[13:]),max(end),max(horizon))])
