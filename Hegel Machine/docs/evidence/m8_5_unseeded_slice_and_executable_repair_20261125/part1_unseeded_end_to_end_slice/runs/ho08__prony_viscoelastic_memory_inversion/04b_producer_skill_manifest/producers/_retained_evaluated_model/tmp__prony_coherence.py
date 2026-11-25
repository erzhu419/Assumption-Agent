import pickle, sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import prony_rheology as m
from prony_stresscheck import clean, strain, probe


def compare(e, truth, G):
    t,x,y,u,z = [e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]
    tr,te = m._History(t,x),m._History(u,z)
    D = np.c_[np.ones(len(t)),(t-t.mean())/np.ptp(t)]
    grid = np.linspace(*m._LOG_BOUNDS,61); A = tr.columns(grid)[0]
    keep,g = m._screen(A,D,y); rows,cols = m._spectrum(tr,grid,A,D,y,keep,g)
    z = m._project(np.linalg.qr(D[keep],mode='reduced')[0],y[keep]); scores = np.array([row[0] for row in rows]); w = np.exp(-.5*(scores-scores.min())); w /= w.sum()
    base = np.zeros(len(u)); coherent = base.copy(); Ghat = 0.
    for k,(weight,(_,v,res)) in enumerate(zip(w,rows)):
        if weight == 0.: continue
        b = m._equilibrium(v,res,cols,keep.sum()); Ghat += weight*b
        pred = te.columns(v[:k])[0]@v[k:]; base += weight*pred
        if k and abs(b-v[k]) > 1e-9 and weight > 1e-9:
            lo,hi = m._LOG_BOUNDS
            def f(p):
                C,_ = cols(p[:k]); return C@np.r_[b,p[k:]]-z
            def j(p):
                C,J = cols(p[:k],True); return np.c_[J*p[k:],C[:,1:]]
            opt = least_squares(f,np.r_[v[:k],v[k+1:]],jac=j,bounds=(np.r_[[lo]*k,[0.]*k],np.r_[[hi]*k,[np.inf]*k]),x_scale='jac',max_nfev=60,ftol=1e-9)
            pred = te.columns(opt.x[:k])[0]@np.r_[b,opt.x[k:]]
        coherent += weight*pred
    scale = np.linalg.norm(truth)
    return np.array([np.linalg.norm(base-truth)/scale,np.linalg.norm(coherent-truth)/scale,abs(Ghat/G-1)])

with open('/holdout_task/public_rheology.pkl','rb') as f: p = pickle.load(f)
print('public [current stress, coherent stress, equilibrium]',compare(p['experiment'],p['gt_test_stress'],p['gt_G_inf']),flush=True)
for name,tau,g,noise in [('fast',[.07],[11.,22.],.015),('short',[.1,.85,4.],[9.,21.,14.,8.],.015),('moderate',[.26,1.7,9.],[10.,21.,14.,8.],.015),('slow',[.63,3.2,16.],[10.,21.,14.,8.],.015),('slow-clean',[.63,3.2,16.],[10.,21.,14.,8.],.0001),('four',[.06,.36,2.1,12.],[14.,15.,13.,11.,8.],.005),('wide',[.12,.5,13.],[7.,35.,10.,12.],.008)]:
    scores=[]
    for i in range(8):
        rng=np.random.default_rng(1190+i); t=np.linspace(0,30,601); u=np.linspace(0,24,481)
        if i%2:
            t[1:-1]+=rng.uniform(-.012,.012,599); u[1:-1]+=rng.uniform(-.018,.018,479)
        x=strain(t); z=probe(u); gg=np.asarray(g); y=clean(t,x,tau,gg); gt=clean(u,z,tau,gg)
        y+=rng.uniform(-1,1)+rng.uniform(-.25,.25)*t+rng.normal(0,noise,len(t))
        ix=rng.choice(len(t),12,replace=False); y[ix]+=rng.choice([-1,1],12)*rng.uniform(.3,2.,12)
        scores.append(compare(dict(train_t=t,train_strain=x,train_stress_raw=y,test_t=u,test_strain=z),gt,g[0]))
    a=np.asarray(scores)
    print(name,'uniform worst',np.max(a[::2],0).round(6),'irregular worst',np.max(a[1::2],0).round(6),'median',np.median(a,0).round(6),flush=True)
