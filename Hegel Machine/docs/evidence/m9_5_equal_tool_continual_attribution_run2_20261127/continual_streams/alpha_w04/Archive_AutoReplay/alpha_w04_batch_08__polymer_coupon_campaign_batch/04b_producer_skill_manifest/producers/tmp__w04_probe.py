import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('prev', '/tmp/hegel_continual_archive/alpha_w04_batch_06_solver.py')
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
d = pickle.load(open('/holdout_task/public_alpha_w04_batch_08.pkl','rb'))[0]['dataset']
a, b = d['train'], d['test']
X, Xt = prev._align(a, b)
dt, t, sy = d['past_dt'], d['eval_times'], d['yield_stress']
obs = a['strain_curves_obs'] - a['strain_0'][:,None]
rate = a['instant_rate_obs']
lab = a['lab_id']
x0, x1 = d['calibration_load_cells']['cmd_stress']; y0, y1 = d['calibration_load_cells']['true_stress']
g0 = (y1-y0)/(x1-x0)
anchors = np.array([[g0, y0-g0*x0], [1.214599, .119990], [1.15, .037965], [.72, -.127660]])
# Candidate clean singleton checks from the historical Wing B sweep.
x_anchor = np.array([-.95, 1.15]); y_anchor = np.array([-1.054535, .700340])

def fit(idx, mode='free', use_plastic=True):
    sub = {k:v[idx] for k,v in a.items()}
    n, k = obs[idx].shape
    target = np.r_[obs[idx].ravel(), rate[idx]]
    p0 = np.r_[np.log([.6, 2.4]), anchors[2:,0]]
    lo = np.r_[np.log([.08,.02]), [.3,.3]]
    hi = np.r_[np.log([5.,16.]), [2.,2.]]
    if mode == 'free':
        p0 = np.r_[p0, anchors[2:,1]]
        lo = np.r_[lo, [-.6,-.6]]; hi = np.r_[hi, [.6,.6]]
    def unpack(p):
        tau = np.cumsum(np.exp(p[:2]))
        affine = anchors.copy()
        affine[2:,0] = p[2:4]
        affine[2:,1] = p[4:6] if mode == 'free' else y_anchor - x_anchor*p[2:4]
        return tau, affine
    def matrix(p):
        tau, affine = unpack(p)
        A, B = prev._design(tau, dt, t, *prev._stress(sub, affine), X[idx], sy)
        return np.vstack([A.reshape(-1,9), B])[:, :9 if use_plastic else 7]
    A = matrix(p0)
    w = np.linalg.lstsq(A,target,rcond=None)[0]
    def run(p, w, scale, loss):
        def f(q): return (matrix(q[:len(p)])@q[len(p):] - target)/scale
        return least_squares(f,np.r_[p,w],bounds=(np.r_[lo,[-np.inf]*len(w)],np.r_[hi,[np.inf]*len(w)]),x_scale='jac',loss=loss,max_nfev=160)
    res = run(p0,w,.04,'soft_l1')
    r = res.fun*.04
    def mad(z): return max(1e-6,1.4826*np.median(np.abs(z-np.median(z))))
    scale = 2.385*np.r_[np.full(n*k,mad(r[:n*k])),np.full(n,mad(r[n*k:]))]
    res = run(res.x[:len(p0)],res.x[len(p0):],scale,'cauchy')
    tau, aff = unpack(res.x[:len(p0)])
    return tau,aff,np.pad(res.x[len(p0):],(0,0 if use_plastic else 2))

def score(idx, model):
    tau, aff, w = model
    sub = {k:v[idx] for k,v in a.items()}
    A,B = prev._design(tau,dt,t,*prev._stress(sub,aff),X[idx],sy)
    C,R=A@w-obs[idx],B@w-rate[idx]
    return np.array([[np.median(np.abs(C[lab[idx]==i])),np.sqrt(np.mean(C[lab[idx]==i]**2)),np.median(np.abs(R[lab[idx]==i])),np.sqrt(np.mean(R[lab[idx]==i]**2))] for i in range(4)])

if __name__ == '__main__':
    rng = np.random.default_rng(613)
    order = rng.permutation(len(lab))
    for mode, use_plastic in [('singleton',True),('free',True),('singleton',False),('free',False)]:
        val=[]
        for test in np.array_split(order,4):
            model=fit(np.setdiff1d(order,test),mode,use_plastic)
            val.append(score(test,model))
            print(mode,use_plastic,'fold',np.round(model[0],5),np.round(model[1][2:],5).tolist(),np.round(model[2][-2:],6))
        model=fit(order,mode,use_plastic)
        print('FULL',mode,use_plastic,*[np.round(q,6) for q in model]); print('OOF per-lab [curve median/RMS, rate median/RMS]:',np.round(np.mean(val,axis=0),6).tolist())
