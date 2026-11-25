from assay_probe import *
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

nodes, weights = hermgauss(12)
nodes, weights = nodes * np.sqrt(2), weights / np.sqrt(np.pi)

def design(z, a, levels, mode):
    miss = ~np.isfinite(a['X'])
    z = z.copy()
    if mode:
        for c in np.unique(a['center_id']):
            rows = a['center_id'] == c
            obs = ~miss[rows]
            counts = obs.sum(axis=0)
            center_mean = (z[rows].sum(axis=0) / np.maximum(counts, 1)).mean()
            means = (z[rows].sum(axis=0) + 24 * center_mean) / (counts + 24)
            z[rows] += miss[rows] * means
    return np.column_stack([z, np.ones(len(z)), 3. * (a['stratum'][:, None] == levels)]), miss.astype(float)

def fit_missing(x,t,z,q,C,mode):
    levels = np.unique(t['stratum'])
    a, am = design(x,t,levels,mode)
    b, bm = design(z,q,levels,mode)
    d = x.shape[1]
    if mode < 2:
        am *= 0; bm *= 0
    y = t['y']
    ridge = np.ones(a.shape[1])/C; ridge[d] = 0.
    def predict(a,m,w):
        sd = np.sqrt(m @ w[:d]**2 + 1e-14)
        u = expit((a @ w)[:,None] + sd[:,None]*nodes)
        return u @ weights, u, sd
    def obj(w):
        p,u,sd = predict(a,am,w)
        p = np.clip(p, 1e-12, 1-1e-12)
        dp = u*(1-u)
        g = (p-y)/(p*(1-p))
        grad = a.T @ (g*(dp@weights))
        grad[:d] += w[:d] * (am.T @ (g*(dp@(weights*nodes))/sd))
        return -np.sum(y*np.log(p)+(1-y)*np.log1p(-p))+.5*np.dot(ridge,w*w), grad+ridge*w
    w=minimize(obj,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7}).x
    return predict(b,bm,w)[0],w[:d]

for mass in (0.,100.,300.):
 for C in (.15,.5,2.):
  for mode in (0,1,2):
    results=[]
    for e in cohorts:
        d,g=e['dataset'],e['gt']; t,q=d['train'],d['query']; r=moments(d['reference_controls']); y=g['y_query']
        x,z=align(t,r,mass),align(q,r,mass)
        p,w=fit_missing(x,t,z,q,C,mode)
        oof=np.zeros(len(x))
        for c in np.unique(t['center_id']):
            m=t['center_id']==c
            oof[m],_=fit_missing(x[~m],subset(t,~m),x[m],subset(t,m),C,mode)
        sc=score(t,t['y'],oof); qs=score(q,y,p)
        results.append(np.round([sc[:,0].min(),sc[:,1].max(),sc[:,1].mean(),qs[:,0].min(),qs[:,1].mean(),np.mean((p-g['prob_query'])**2)],4).tolist())
    print(mass,C,mode,*results)
