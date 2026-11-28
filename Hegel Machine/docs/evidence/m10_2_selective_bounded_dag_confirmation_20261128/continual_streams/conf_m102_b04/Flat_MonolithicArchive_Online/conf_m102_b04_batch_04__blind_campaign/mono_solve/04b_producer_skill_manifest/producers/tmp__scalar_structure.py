exec(open('/tmp/stress_batch.py').read().split('for mode in')[0])
from scipy.optimize import minimize
from itertools import combinations

def fit(g,y,ridge=None):
    if ridge is None: return np.linalg.lstsq(g,y,rcond=None)[0]
    return np.linalg.lstsq(np.vstack((g,np.diag(np.sqrt(ridge)))),np.r_[y,np.zeros(len(ridge))],rcond=None)[0]

E=[]
for name,i,j in folds:
    T=m._time_basis(s['eval_times'],z[i]); v=z[i]@T
    p=m._profile(u[i],np.c_[y[i],v]); q=m._quadratic(u); g=m._design(u,q,p)
    co=np.linalg.lstsq(g[i],v,rcond=None)[0]; a=g@co
    h=g[:,15:]; h=h/np.linalg.norm(h[i],axis=0)
    # The measured state contains a stable common nonlinearity. Compare scalar readout laws that retain it.
    odd=np.c_[q[:,:5],h]
    av=a / np.std(a[i],axis=0)
    cs=[av[:,k]*av[:,l] for k,l in [(0,0),(0,1),(1,1)]]
    candidates={'base':g,'odd':odd,'diag':np.c_[odd,q[:,5:9]],'mixed':np.c_[odd,q[:,9:]],'statequad':np.c_[g,*cs], 'oddcubic':np.c_[g,u**3], 'stateodd':np.c_[q,av], 'statecross':np.c_[g,*cs,*[u[:,k]*av[:,l] for k in range(4) for l in range(2)]]}
    v=np.zeros_like(u); w=abs(u); c,k,r=np.exp(p); v=np.sign(u)*np.exp(k*np.log(np.maximum(w,1e-30))-r*np.log1p(c*w*w));
    for power in [1.5,2,3]:
        for weight in [0.1,0.4,1.6]:
            hh=np.sign(u)*(w**power); gg=np.c_[q,h,hh]; penalty=np.r_[0,np.repeat(.03,4),np.repeat(.25,10),np.zeros(4),np.repeat(weight,4)]; candidates[(power,weight)]=(gg,penalty)
    err=[]
    for key,gg in candidates.items():
        ridge=None
        if isinstance(gg,tuple): gg,ridge=gg
        c0=fit(gg[i],y[i],ridge); err.append(abs(gg[j]@c0-y[j]).mean())
    # Lagged data are used only as an explanatory check, not as unavailable prediction inputs.
    if name=='52/20':
        print('y summaries residual RMS',np.std(y[i]-g[i]@fit(g[i],y[i])), 'rank',np.linalg.svd(np.c_[fit(g[i],y[i])[-4:],co[-4:]],compute_uv=False))
        print('state loadings across axes and nonlinear y/state coefficients',np.c_[fit(g[i],y[i])[-4:],co[-4:]])
    E.append(err); print(name, np.round(err,5))
print(list(candidates)); e=np.array(E);print('worst',e.max(0).round(5));print('average',e.mean(0).round(5));print('ordinary',e[:9].mean(0).round(5))
