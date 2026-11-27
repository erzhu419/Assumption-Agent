import itertools, pickle, runpy
import numpy as np

root = '/holdout_task/'
s = runpy.run_path(root+'alpha_m96_w03_s00_solver.py')
e, = pickle.load(open(root+'public_alpha_m96_w03_s00.pkl','rb'))
d = e['dataset']; c = d['calibration_load_cells']; base = s['_calibrate'](c)

def fit_line(x, y, command_limits, yield_stress, cutoff):
    a, b = np.triu_indices(len(x), 1)
    separated = np.abs(x[a]-x[b]) > 1e-10
    a, b = a[separated], b[separated]
    if not len(a):
        return np.array([1., np.median(y-x) if len(x) else 0.])
    gains = (y[a]-y[b])/(x[a]-x[b]); biases = y[a]-gains*x[a]
    inliers = np.abs(y[None,:] - gains[:,None]*x - biases[:,None]) <= cutoff
    design = np.column_stack((x, np.ones(len(x))))
    fits = []
    for mask in np.unique(inliers,axis=0):
        if np.ptp(x[mask]) <= 1e-10: continue
        fit = np.linalg.lstsq(design[mask],y[mask],rcond=None)[0]
        error = y[mask]-design[mask]@fit
        exceedance = max(0.,np.max(abs(fit[0]*command_limits+fit[1]))-yield_stress-cutoff)
        fits.append((exceedance, -mask.sum(), np.ptp(x[mask]), error@error, fit))
    if not fits:
        return np.linalg.lstsq(design,y,rcond=None)[0]
    return fits

for i in range(4):
    m = c['lab_id']==i; x,y = c['cmd_stress'][m],c['true_stress'][m]; g0,b0 = (p[i] for p in base)
    physical = np.concatenate([d[k][f][d[k]['lab_id']==i].ravel() for k in ('train','test') for f in ('past_stress_cmd','stress_hold_cmd')])*g0+b0
    res = [[],[],[],[]]; witness = [None]*4
    for g,b in [(g0,b0),(.86+.085*i,.11-.08*i)]:
        xx = [x[list(ids)] for ids in itertools.combinations(range(len(x)),4)]; yy = y+(g-g0)*x+b-b0
        limits = (np.array([physical.min(),physical.max()])-b)/g
        for ids, xs in zip(itertools.combinations(range(len(x)),4),xx):
            if np.ptp(xs) < .7*np.ptp(x): continue
            for j in range(4):
                for spike in (-.8,.8):
                    ys=yy[list(ids)].copy(); ys[j]+=spike
                    for k,cutoff in enumerate((.00035,.0006)):
                        fits=fit_line(xs,ys,limits,d['yield_stress'],cutoff)
                        for t in (0,1):
                            u=2*k+t
                            best=min(fits,key=(lambda v:(v[0],v[1],v[3])) if t==0 else (lambda v:(v[0],v[1],-v[2],v[3])))
                            a0,b0_=best[-1]; error=np.max(abs((a0-g)*x+b0_-b))
                            if not res[u] or error>max(res[u]): witness[u]=(xs.tolist(),j,spike,best[:4])
                            res[u].append(error)
    print('lab',i,'yield',d['yield_stress'],'observed max',max(abs(physical)))
    for u,r in enumerate(res): print(u,'median/p95/max',np.quantile(r,[.5,.95,1]),'worst',witness[u])
