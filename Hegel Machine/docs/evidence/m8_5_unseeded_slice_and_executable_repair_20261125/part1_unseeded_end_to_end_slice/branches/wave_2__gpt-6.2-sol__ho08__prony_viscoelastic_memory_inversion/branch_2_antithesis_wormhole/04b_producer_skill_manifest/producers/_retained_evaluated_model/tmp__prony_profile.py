import numpy as np
from scipy.optimize import least_squares
import prony_sparse as s
from prony_bias import local, problem


def profile(t, eps, y, keep, v, rms, ngrid=31):
    m=(len(v)-1)//2
    proj,cols=problem(t,eps,keep); z=proj(y[keep]); lo,hi=np.log([.02,20.]); nxt=np.delete(v,m-1)
    lb=np.r_[[lo]*(m-1),[0.]*(m+1)]; ub=np.r_[[hi]*(m-1),[np.inf]*(m+1)]
    # Concentrate integration near the fitted time while covering the full admissible slow tail.
    zz=np.unique(np.r_[np.linspace(max(lo, v[m-2]),hi,ngrid),v[m-1]])
    rows=[]
    for k in zz:
        def fun(w):
            X,_=cols(np.r_[w[:m-1],k]); return X@w[m-1:]-z
        def jac(w):
            X,H=cols(np.r_[w[:m-1],k],True); return np.c_[H[:,:m-1]*w[m:-1],X]
        up=ub.copy(); up[:m-1]=k
        w=np.minimum(up-1e-9,np.maximum(lb+1e-9,nxt))
        f=least_squares(fun,w,jac=jac,bounds=(lb,up),max_nfev=40,x_scale='jac',ftol=1e-8)
        sv=np.linalg.svd(f.jac,compute_uv=False); vol=-np.log(np.maximum(sv,sv[0]*1e-6)).sum()
        rows.append([f.cost,f.x[m-1],vol])
    r=np.array(rows); dx=np.r_[np.diff(zz),0]+np.r_[0,np.diff(zz)]; base=-(r[:,0]-r[:,0].min())/rms**2+np.log(dx)
    def average(logw):
        w=np.exp(logw-logw.max()); return w@r[:,1]/w.sum()
    return [average(base+a*r[:,2]+b*zz) for a,b in [(0,0),(1,0),(0,-1),(1,-1),(0,1)]]


if __name__=='__main__':
    t,eps,y=s.t,s.eps,s.y; n=len(t); m=3
    tau,g,b,rss,keep=s.sparse(np.ones(n,bool),m,starts=5)
    v,vc,sd,cost=local(t,eps,y,keep,np.r_[np.log(tau),g]); rms=np.sqrt(2*cost/(keep.sum()-9))
    print('public truth',s.G,'raw/corrected',v[m],vc[m],'profile',profile(t,eps,y,keep,v,rms,51),flush=True)
    rng=np.random.default_rng(9183)
    for tt in ([.09,.8,4.],[.25,1.7,9.],[.6,3.,16.],[.08,.5,17.]):
        gg=np.array([10.,21.,14.,8.]); y0=s.memory(t,eps,np.array(tt))@gg+s.D@np.array([.2,-3.]); start=np.r_[np.log(tt),gg]; rows=[]
        for k in range(24):
            keep=np.ones(n,bool); keep[rng.choice(n,12,replace=False)]=False
            yy=y0+rng.normal(0,.015,n); u,uc,std,c=local(t,eps,yy,keep,start)
            rows.append([u[m],uc[m],*profile(t,eps,yy,keep,u,np.sqrt(2*c/(keep.sum()-9)),35)])
        er=np.array(rows)/gg[0]-1
        print('synthetic',tt,'bias',er.mean(0).round(4),'q90',np.quantile(abs(er),.9,axis=0).round(4),'worst',abs(er).max(0).round(4),'pass',np.mean(abs(er)<=.06,axis=0).round(3),flush=True)
