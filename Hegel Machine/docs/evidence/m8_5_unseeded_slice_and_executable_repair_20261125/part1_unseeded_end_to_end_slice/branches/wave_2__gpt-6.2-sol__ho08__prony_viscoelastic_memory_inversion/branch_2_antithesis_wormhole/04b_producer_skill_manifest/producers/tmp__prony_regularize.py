import pickle, sys
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import prony_rheology as mod
from prony_stresscheck import strain, probe, clean

# Regularize poorly constrained relative spectral geometry, not the modulus or absolute time.
def refit(hist, t, y, keep, v0, strength, shape):
    m=(len(v0)-1)//2
    if m==0: return v0
    v0=np.r_[v0[:m][np.argsort(v0[:m])],v0[m],v0[m+1:][np.argsort(v0[:m])]]
    D=np.c_[np.ones_like(t),(t-t.mean())/np.ptp(t)]
    q=np.linalg.qr(D[keep], mode='reduced')[0]; P=lambda a:mod._project(q,a); z=P(y[keep]); lo,hi=mod._LOG_BOUNDS
    C=hist.columns(v0[:m])[0]; r=P(C[keep])@v0[m:]-z
    noise=np.sqrt(np.mean(r*r)); amp_floor=.05*max(np.max(v0[m+1:]),1e-8)
    F=np.diff(np.eye(m),axis=0); S=np.diff(F,axis=0)
    def terms(v):
        if shape=='spacing': return S@v[:m]
        if shape=='amplitude': return F@np.log(v[m+1:]+amp_floor)
        return np.r_[S@v[:m],F@np.log(v[m+1:]+amp_floor)]
    def residual(v):
        C=hist.columns(v[:m])[0]; return np.r_[P(C[keep])@v[m:]-z,noise*strength*terms(v)]
    def jac(v):
        C,H=hist.columns(v[:m],True); J=np.c_[P(H[keep])*v[m+1:],P(C[keep])]
        u=np.c_[S,np.zeros((len(S),m+1))]; w=np.c_[np.zeros((len(F),m+1)),F/(v[m+1:]+amp_floor)]
        return np.vstack([J,noise*strength*({'spacing':u,'amplitude':w,'joint':np.vstack([u,w])}[shape])])
    if not len(terms(v0)): return v0
    l=np.r_[[lo]*m,[0.]*(m+1)]; h=np.r_[[hi]*m,[np.inf]*(m+1)]
    fit=least_squares(residual,np.clip(v0,l+1e-10,h-1e-10),jac=jac,bounds=(l,h),max_nfev=120,ftol=1e-9,x_scale='jac')
    return fit.x

SPECS=[('base',0,''),('space1',1.,'spacing'),('space2',2.,'spacing'),('amp1',1.,'amplitude'),('amp2',2.,'amplitude'),('both1',1.,'joint'),('both2',2.,'joint')]
def assess(e, gt, G):
    t,x,y,u,z=[e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]
    hist,test=mod._History(t,x), mod._History(u,z); D=np.c_[np.ones_like(t),(t-t.mean())/np.ptp(t)]
    grid=np.linspace(*mod._LOG_BOUNDS,61); A=hist.columns(grid)[0]; k,g=mod._screen(A,D,y)
    v,res,columns=mod._spectrum(hist,grid,A,D,y,k,g); m=(len(v)-1)//2
    Gbase=mod._equilibrium(v,res,columns,k.sum()); rows=[]
    for _,strength,shape in SPECS:
        w=refit(hist,t,y,k,v,strength,shape) if strength else v
        # Evaluate direct regularization and its midpoint with the bias-corrected estimator.
        yp=test.columns(w[:m])[0]@w[m:]; Gw=w[m] if strength else Gbase
        rows.append([np.linalg.norm(yp-gt)/np.linalg.norm(gt),abs(Gw/G-1),abs((Gw+Gbase)/2/G-1)])
    return np.array(rows)

if __name__=='__main__':
    with open('/holdout_task/public_rheology.pkl','rb') as f: p=pickle.load(f)
    print('models', [s[0] for s in SPECS],flush=True)
    print('public stress,G,Gmid',assess(p['experiment'],p['gt_test_stress'],p['gt_G_inf']).T.round(5),flush=True)
    for name,tau,g,noise in [('single',[.07],[11.,22.],.015),('short',[.10,.85,4.],[9.,21.,14.,8.],.015),('moderate',[.26,1.7,9.],[10.,21.,14.,8.],.015),('slow',[.63,3.2,16.],[10.,21.,14.,8.],.015),('slow-clean',[.63,3.2,16.],[10.,21.,14.,8.],.0001),('four',[.06,.36,2.1,12.],[14.,15.,13.,11.,8.],.005),('uneven',[.04,1.8,5.],[8.,9.,19.,6.],.012)]:
        rows=[]
        for j in range(8):
            rng=np.random.default_rng(1190+j); t=np.linspace(0,30,601); u=np.linspace(0,24,481)
            if j%2:
                t[1:-1]+=rng.uniform(-.012,.012,599); u[1:-1]+=rng.uniform(-.018,.018,479)
            x=strain(t); z=probe(u); gg=np.array(g); y=clean(t,x,tau,gg); gt=clean(u,z,tau,gg)
            y+=rng.uniform(-1,1)+rng.uniform(-.25,.25)*t+rng.normal(0,noise,len(t))
            ix=rng.choice(len(t),12,replace=False); y[ix]+=rng.choice([-1,1],12)*rng.uniform(.3,2.,12)
            rows.append(assess(dict(train_t=t,train_strain=x,train_stress_raw=y,test_t=u,test_strain=z),gt,g[0]))
        r=np.array(rows)
        print(name, 'worst stress,G,Gmid',r.max(0).T.round(4).tolist(), 'median',np.median(r,axis=0).T.round(4).tolist(),flush=True)
