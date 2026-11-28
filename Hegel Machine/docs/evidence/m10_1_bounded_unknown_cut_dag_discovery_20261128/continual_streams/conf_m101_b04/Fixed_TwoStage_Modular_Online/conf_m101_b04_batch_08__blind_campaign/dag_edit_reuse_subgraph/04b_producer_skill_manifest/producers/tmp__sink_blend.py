from sink_probe import *
import runpy
mod = runpy.run_path('/holdout_task/node_01.py')

def kernel(x, w, y, shape, gaussian=False, affine=False):
    dd = ((x[:,None]-x[None,:])**2).sum(2)
    dv = ((w[:,None]-x[None,:])**2).sum(2)
    scale = shape*np.median(dd[np.triu_indices(len(x),1)])
    fun = (lambda q: np.exp(-q/2)) if gaussian else (lambda q: (1+q/4)**-2)
    B = np.c_[np.ones(len(x)),x] if affine else np.ones((len(x),1))
    Bt = np.c_[np.ones(len(w)),w] if affine else np.ones((len(w),1))
    A = fun(dd/scale) + 0.0003*np.eye(len(x))
    P = np.linalg.solve(A,B)
    R = np.linalg.solve(A,y)
    c = np.linalg.solve(B.T@P,B.T@R)
    return Bt@c+fun(dv/scale)@(R-P@c)

if __name__ == '__main__':
    groups = [[np.arange(52,N)], list(np.array_split(np.random.default_rng(30).permutation(N),6)), list(np.array_split(np.random.default_rng(117).permutation(N),6)), list(np.array_split(np.arange(N),6))]
    configs = [(.65,.85),(.75,1.1),(.8,1.3),(.8,1.6),(.9,1.3),(1.,1.3)]
    metrics = [[] for _ in configs]; traj = []
    for folds in groups:
        rs = []; ts = []
        for v in folds:
            a = np.setdiff1d(np.arange(N),v); xa=d['xa']; xb=d['xab']; z=d['z']
            b = mod['_predict'](xa[a],xa[v],xb[a],xb[v],Y[a,0],1.8,.1)
            t = kernel(z[a],z[v],Y[a,1:],1.3,True,True)
            tb = mod['_predict'](z[a],z[v],xb[a],xb[v],Y[a,1:],.6,.01)
            ts.append([np.abs(p-Y[v,1:]).mean() for p in (tb,t,.25*tb+.75*t)]+[np.abs(t-Y[v,1:]).mean(0).max()])
            rs.append([np.mean(np.abs((1-f)*b+f*kernel(xa[a],xa[v],Y[a,0],h)-Y[v,0])) for f,h in configs]+[np.abs(b-Y[v,0]).mean()])
        rs=np.array(rs); ts=np.array(ts)
        print('scalar mean/max', np.round(rs.mean(0),5), np.round(rs.max(0),5));print('traj mean/max (baseline kernel blend kernel_worst_horizon)',np.round(ts.mean(0),5),np.round(ts.max(0),5));print('configs',configs)
