import pickle
import numpy as np
from scipy.optimize import least_squares
from prony_probe import memory, solve

p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']; t=e['train_t']; eps=e['train_strain']; y=e['train_stress_raw']; te=e['test_t']; et=e['test_strain']; yt=p['gt_test_stress']; G=p['gt_G_inf']
D=np.c_[np.ones_like(t),(t-t.mean())/np.ptp(t)]; grid=np.geomspace(.02,20,61); A=memory(t,eps,grid)

def sparse(keep,m,seed=0,starts=9):
    visible=keep.copy(); g,b,s,r=solve(A[keep],D[keep],y[keep],'clip'); keep[visible] = abs(r) < 4*s
    T=t[keep]; X=D[keep]; Y=y[keep]; Q=np.linalg.qr(X)[0]; z=Y-Q@(Q.T@Y)
    def design(logtau):
        C=memory(t,eps,np.exp(logtau))[keep]; return C-Q@(Q.T@C)
    def residual(v):
        return design(v[:m])@v[m:]-z
    def jac(v):
        h=1e-4; C=design(v[:m]); J=(design(v[:m]+h)[:,1:]-design(v[:m]-h)[:,1:])/(2*h)*v[m+1:]; return np.c_[J,C]
    rng=np.random.default_rng(seed); lo,hi=np.log([.02,20]); power=g[1:]*np.linalg.norm(A[visible,1:],axis=0); weights=power/(power.sum() or 1); cdf=np.cumsum(weights); selected=np.interp((np.arange(m)+.5)/m,cdf,np.log(grid)); base=np.linspace(np.log(.12),np.log(10),m) if m>1 else np.array([0.])
    best=None
    for j in range(starts):
        init=base if j==0 else selected if j==1 else np.sort(np.clip(base+rng.normal(0,.9,m),lo+.005,hi-.005))
        init=np.clip(init,lo+.005,hi-.005)
        from scipy.optimize import nnls
        h=nnls(design(init),z)[0]; v=np.r_[init,np.maximum(h,.01)]
        f=least_squares(residual,v,jac=jac,bounds=(np.r_[[lo]*m,[0]*(m+1)],np.r_[[hi]*m,[np.inf]*(m+1)]),x_scale='jac',max_nfev=180,ftol=1e-8)
        if best is None or f.cost<best.cost: best=f
    v=best.x; tau=np.exp(v[:m]); g=v[m:]; b=np.linalg.lstsq(X,Y-memory(t,eps,tau)[keep]@g,rcond=None)[0]; return tau,g,b,2*best.cost,keep

if __name__=='__main__':
    for j,ix in enumerate([np.array([],int)]+list(np.array_split(np.arange(len(t)),6))):
        keep=np.ones(len(t),bool); keep[ix]=False; rows=[]
        for m in range(1,5):
            tau,g,b,rss,k=sparse(keep.copy(),m,starts=9 if j==0 else 5); pred=memory(te,et,tau)@g; err=np.linalg.norm(pred-yt)/np.linalg.norm(yt); residual=y[ix]-memory(t,eps,tau)[ix]@g-D[ix]@b; val=np.median(abs(residual)) if len(ix) else 0.; bic=k.sum()*np.log(rss/k.sum())+(2*m+3)*np.log(k.sum()); rows.append((m,round(bic,2),round(val,5),round(err,6),round(g[0],4),tau.round(3).tolist(),g[1:].round(2).tolist()))
        print('fold',j,'rows (M,BIC,median abs held residual,test rel,G,tau,amps)',rows,flush=True)
