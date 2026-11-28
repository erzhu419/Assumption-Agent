import itertools
import pickle
import runpy
import numpy as np
m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
sw = s['sensor_calibration_sweeps']
x, y = sw['ref_rosette'], sw['raw_rosette']
single = sw['single_bank_sweeps']
left, right = np.roll(x,1,axis=1), np.roll(x,-1,axis=1)
f = np.stack((np.ones_like(x), x,x**3,left,right,x*right),axis=-1)
f1 = []
y1 = []
for i in range(4):
    j = single['bank_id']==i
    t = single['ref_signal'][j]
    z = np.zeros_like(t)
    f1.append(np.column_stack((1+z,t,t**3,z,z,z)))
    y1.append(single['raw_readout'][j])

def design(base, extra):
    _, a, a3, b, c, ac = base.T
    if extra == 'pairwise':
        return np.column_stack((base, a*b, b*c, (a-c)**3-a3+c**3))
    if extra == 'rational':
        q = (a-c)**2
        return np.column_stack((base, a*q/(1+q), ac*q/(1+q)))
    return base

def fits(fs, ys, extra):
    d = [design(a,extra) for a in fs]
    co = np.array([m['_fit_sweep'](a,b) for a,b in zip(d,ys)])
    if extra != 'shared':
        return co
    keep = [abs(a@c-b)<1e-4 for a,c,b in zip(d,co,ys)]
    d = [a[k] for a,k in zip(d,keep)]
    ys = [b[k] for b,k in zip(ys,keep)]
    ratios = (co[:,2:]/co[:,1,None]).mean(0)
    for _ in range(12):
        og = np.array([np.linalg.lstsq(np.column_stack((a[:,0],a[:,1]+a[:,2:]@ratios)),b,rcond=None)[0] for a,b in zip(d,ys)])
        ratios = np.linalg.lstsq(np.vstack([p[1]*a[:,2:] for p,a in zip(og,d)]),np.concatenate([b-a[:,:2]@p for a,b,p in zip(d,ys,og)]),rcond=None)[0]
    return np.column_stack((og,og[:,1,None]*ratios))

folds = np.array_split(np.random.default_rng(820).permutation(len(x)), 5)
for extra in ('cyclic','pairwise','rational','shared'):
    errors = []
    for test in folds:
        train = np.setdiff1d(np.arange(len(x)),test)
        augx = [np.vstack((f[train,i],f1[i])) for i in range(4)]
        augy = [np.r_[y[train,i],y1[i]] for i in range(4)]
        co = fits(augx,augy,extra)
        errors.append(abs(np.column_stack([design(f[test,i],extra)@co[i] for i in range(4)])-y[test]))
    pooled = np.vstack(errors)
    print(extra,'median/q95/worst clean',np.quantile(pooled[pooled<1],[.5,.95,1]),'worst clean per bank',[max(a[a<1]) for a in pooled.T],'spikes',sum((pooled>=1).ravel()))

cal = m['calibrate'](s)
z = s['Z_nuisance']; ns = len(s['sensor_ij'])
roi = np.arange(len(cal)) % ns // 4
shift = np.arange(len(cal)) // ns
base = np.repeat([s['robin_x'],s['robin_y']],2)
for pattern in [(0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(1,1,1,1)]:
    S,_ = m['chamber_response'](s,base*np.array(pattern))
    for use_z in ([True,False] if not any(pattern) else [True]):
        report = []
        for groups in (shift,roi):
            errors = []
            for k in np.unique(groups):
                j = groups!=k
                zz = z if use_z else z[:,:0]
                w,_ = m['_inverse']({**s,'Z_nuisance':zz[j]},S[j],cal[j])
                Z = zz[j]
                gam = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@(cal[j]-S[j]@w))
                errors.append(np.sqrt(np.mean((cal[~j]-S[~j]@w-zz[~j]@gam)**2)))
            report.append((max(errors),np.mean(errors)))
        print('walls',pattern,'nuisance',use_z,'worst/mean by shift and sensor group',report)
