import pickle
import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter


def memory(t, strain, tau):
    dt = np.diff(t)
    de = np.diff(strain)
    x = dt[:, None] / tau
    drive = -np.expm1(-x) / x * de[:, None]
    q = np.zeros((len(t), len(tau)))
    if np.allclose(dt, dt[0], rtol=1e-10, atol=0):
        for j, a in enumerate(np.exp(-dt[0] / tau)):
            q[1:, j] = lfilter([1.], [1., -a], drive[:, j])
    else:
        decay = np.exp(-x)
        for i in range(len(dt)):
            q[i+1] = decay[i] * q[i] + drive[i]
    return np.column_stack((strain, q))


def solve(A, D, y, mode='cauchy'):
    w = np.ones(len(y))
    for it in range(1 if mode == 'ols' else 18):
        Q, R = np.linalg.qr(D * w[:, None], mode='reduced')
        X, z = A*w[:,None], y*w
        U, v = Q.T@X, Q.T@z
        X, z = X-Q@U, z-Q@v
        norm = np.linalg.norm(X, axis=0)
        norm[norm == 0] = 1
        g = nnls(X/norm, z, maxiter=20*A.shape[1], atol=1e-10)[0] / norm
        b = np.linalg.solve(R, v-U@g)
        r = y-A@g-D@b
        s = max(1e-12, 1.4826*np.median(np.abs(r-np.median(r))))
        if mode == 'cauchy':
            w = 1/np.hypot(1,r/s)
        else:
            w = (np.abs(r) < 4*s).astype(float)
    return g,b,s,r


if __name__ == '__main__':
    p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']; t=e['train_t']; y=e['train_stress_raw']; te=e['test_t']; et=e['test_strain']; eps=e['train_strain']; ys=p['gt_test_stress']; G=p['gt_G_inf']
    print('strain waveform t/eps:',np.c_[t[::20],eps[::20]].round(5).tolist())
    D=np.column_stack((np.ones_like(t), (t-t.mean())/np.ptp(t)))
    for n in (31,61,101):
        tau=np.geomspace(.02,20,n); A=memory(t,eps,tau); B=memory(te,et,tau)
        for mode in ['ols','cauchy','clip']:
            g,b,s,r=solve(A,D,y,mode)
            print(n,mode,'rMAD',round(s,5),'spikes',sum(abs(r)>4*s),'b',b.round(4),'G',round(g[0],5),'test rel',round(np.linalg.norm(B@g-ys)/np.linalg.norm(ys),6),'active',list(zip(tau[g[1:]>0].round(4),g[1:][g[1:]>0].round(3))))
    n=61; tau=np.geomspace(.02,20,n); A=memory(t,eps,tau); B=memory(te,et,tau)
    for j,ix in enumerate(np.array_split(np.arange(len(t)),6)):
        k=np.setdiff1d(np.arange(len(t)),ix); g,b,s,r=solve(A[k],D[k],y[k],'clip')
        err=y[ix]-A[ix]@g-D[ix]@b
        print('blocked',j,'err median',round(np.median(abs(err)),5),'G rel',round(abs(g[0]/G-1),6),'test rel',round(np.linalg.norm(B@g-ys)/np.linalg.norm(ys),6))
