import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
import prony_sparse as s

def problem(t,eps,keep):
    dt=np.diff(t); dx=np.diff(eps); uniform=np.allclose(dt,dt[0],rtol=1e-10,atol=0); ds=dt[0] if uniform else dt
    D=np.c_[np.ones(len(t)),(t-t.mean())/np.ptp(t)][keep]; Q=np.linalg.qr(D)[0]
    def proj(a): return a-Q@(Q.T@a)
    def columns(logs, derivative=False):
        K=np.empty((len(t),1+len(logs))); K[:,0]=eps
        H=np.zeros((len(t),len(logs))) if derivative else None
        for j,log in enumerate(logs):
            x=ds*np.exp(-log); a=np.exp(-x); c=-np.expm1(-x)/x
            q=K[:,j+1]; q[0]=0.
            if uniform:
                q[1:]=lfilter([c],[1.,-a],dx)
                if derivative: H[1:,j]=lfilter([1.],[1.,-a],x*a*q[:-1]+(c-a)*dx)
            else:
                for i in range(len(dx)):
                    q[i+1]=a[i]*q[i]+c[i]*dx[i]
                    if derivative: H[i+1,j]=a[i]*(H[i,j]+x[i]*q[i])+(c[i]-a[i])*dx[i]
        return proj(K[keep]),proj(H[keep]) if derivative else None
    return proj,columns

def local(t,eps,y,keep,v0):
    m=(len(v0)-1)//2; proj,columns=problem(t,eps,keep); z=proj(y[keep]); lo=np.r_[[np.log(.02)]*m,[0.]*(m+1)]; hi=np.r_[[np.log(20.)]*m,[np.inf]*(m+1)]
    def f(v): return columns(v[:m])[0]@v[m:]-z
    def jac(v):
        A,H=columns(v[:m],True)
        return np.c_[H*v[m+1:],A]
    v0=np.maximum(lo+1e-9,np.minimum(hi-1e-9,v0))
    res=least_squares(f,v0,jac=jac,bounds=(lo,hi),x_scale='jac',max_nfev=160,ftol=1e-10)
    v=res.x; J=res.jac; P=np.linalg.pinv(J,rcond=1e-9); C=(P@P.T)*(2*res.cost/max(1,len(z)-len(v)-2)); A,H=columns(v[:m],True); h=1e-4
    H2=(columns(v[:m]+h,True)[1]-columns(v[:m]-h,True)[1])/(2*h)
    trace=H2@(v[m+1:]*np.diag(C)[:m])+2*H@C[np.arange(m),m+1+np.arange(m)]
    corrected=v+.5*(P@trace)
    err=np.sqrt(C[m,m]); return v,corrected,err,res.cost

if __name__=='__main__':
    t,eps,y=s.t,s.eps,s.y; n=len(t); m=3
    tau,g,b,_,keep=s.sparse(np.ones(n,bool),m,starts=5); v0=np.r_[np.log(tau),g]; v,vc,sd,cost=local(t,eps,y,keep,v0)
    print('public G, bias-corrected G, standard error, taus:',v[m],vc[m],sd,np.exp(v[:m]),flush=True)
    for k in (2,4,8):
        vs=np.array([local(t,eps,y,keep & (np.arange(n)%k!=j),v)[0][m] for j in range(k)])
        print('interleaved jackknife',k,vs,'corrected',k*v[m]-(k-1)*vs.mean())
    rng=np.random.default_rng(2916)
    # Synthetic spectra in three separation/slow-tail regimes; no clean public targets used.
    for tt in ([.09,.8,4.],[.25,1.7,9.],[.6,3.,16.],[.08,.5,17.]):
        gg=np.array([10.,21.,14.,8.]); y0=s.memory(t,eps,np.array(tt))@gg+s.D@np.array([.2,-3.]); start=np.r_[np.log(tt),gg]
        rows=[]
        for k in range(90):
            yy=y0+rng.normal(0,.015,n); u,uc,std,_=local(t,eps,yy,np.ones(n,bool),start); rows.append([u[m],uc[m],std,np.exp(u[m-1])]);
        a=np.array(rows); e=a[:,:2]/gg[0]-1
        print('synthetic',tt,'bias',e.mean(0),'median',np.median(e,axis=0),'q90',np.quantile(abs(e),.9,axis=0),'pass',np.mean(abs(e)<=.06,axis=0),'max',abs(e).max(0),'median_se',np.median(a[:,2]),'slow_max',max(a[:,3]),flush=True)
