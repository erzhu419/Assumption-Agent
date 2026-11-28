import importlib.util, pickle
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
x, y, T = (s['train'][k] for k in ('x','y','traj'))
t = s['eval_times']
print('study keys', sorted(s), 'train keys', sorted(s['train']), 'times', t)
a = s['panel_a']; vhat = m._calibrate(a, a['v']); print('calibration inverse row errors', np.quantile(np.linalg.norm(vhat-a['u'],axis=1),[.5,.8,.9,1]))
z,b,q,c,h,d,k = m._features(s,x)
F,G = m._designs(s,x)
base = np.column_stack((m._affine(q),h,m._dot(b,c,2),m._dot(z,c)))
diag = np.column_stack((base,q*h))
compact_y = F[:, [*range(9),14]]
pairs = np.column_stack([z[:,i]+sg*z[:,j] for i,j in combinations(range(4),2) for sg in (-1,1)])
rat = 0.5
ys = {'baseline': F, 'compact': compact_y, 'paired': np.c_[compact_y,np.cos(k*pairs),np.sin(k*pairs)], 'rational':np.c_[compact_y,compact_y[:,1:]/(1+rat*m._dot(c,d)[:,None]**2)]}
gs = {'baseline': G, 'diagonal': diag, 'paired':np.c_[diag,np.cos(k*pairs),np.sin(k*pairs)], 'rational':np.c_[diag,diag[:,1:]/(1+rat*m._dot(c,d)[:,None]**2)]}
rows = np.arange(len(x))
outer = [('52/20',rows[52:]),('56/16',rows[56:]),('edge',np.argsort(np.max(abs(z),axis=1))[-16:])]
outer += [(f'fold{i}',v) for i,v in enumerate(np.array_split(np.random.default_rng(19).permutation(rows),4))]

def state_error(f, a, b):
    def basis(r): return -np.expm1(-t[:,None]*r)
    coef = m._fit(f[a], T[a]); pred = f[a] @ coef
    def residual(r):
        H = basis(r)
        return (H @ m._fit(H,pred.T)-pred.T).ravel()
    H = basis(least_squares(residual,[.5,2.],bounds=(.01,10.)).x)
    amps = m._fit(H,coef.T).T
    return abs(f[b]@amps@H.T - T[b])

def scalar_error(f,a,b):
    return abs(f[b]@m._fit(f[a],y[a])-y[b])

for label,outer_b in outer:
    outer_a=np.setdiff1d(rows,outer_b)
    chunks=np.array_split(np.random.default_rng(129).permutation(outer_a),4)
    folds=[(np.setdiff1d(outer_a,b),b) for b in chunks]
    for name, fs, err in [('scalar',ys,scalar_error),('state',gs,state_error)]:
        inner={k:np.mean([err(f,a,b).mean() for a,b in folds]) for k,f in fs.items()}
        winner=min(inner,key=inner.get)
        tested={k:err(f,outer_a,outer_b) for k,f in fs.items()}
        if name=='state':
            H=m._time_modes(t,T[outer_a]); A=m._fit(H,T[outer_a].T).T
            tested['deployed']=abs(G[outer_b]@m._fit(G[outer_a],A)@H.T-T[outer_b]); print(label,'rate-only reconstruction',round(np.mean(abs(A@H.T-T[outer_a])),6))
        print(label,name,'selected',winner,'inner', {k:round(v,6) for k,v in inner.items()},'outer mean/worst-time', {k:(round(v.mean(),6),round(np.max(v.mean(axis=0)),6)) for k,v in tested.items()})
