"""Calibrated regression with separate readout and temporal response kernels."""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import cho_factor, cho_solve


def fit(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-11)[0]


def affine(x):
    return np.c_[np.ones(len(x)),x]


def calibrate(panel,x):
    u,v=[np.asarray(panel[k],float) for k in ('u','v')]
    good=np.linalg.norm(v,axis=1)<5
    def design(u,j):
        a=u[:,j]; b=u[:,(j+1)%4]
        return np.c_[affine(u),a*b,a**3]
    co=np.column_stack([fit(design(u[good],j),v[good,j]) for j in range(4)])
    def forward(z):
        return np.column_stack([design(z,j)@co[:,j] for j in range(4)])
    g=np.linalg.solve(co[1:5].T,(x-co[0]).T).T
    return np.array([least_squares(lambda z:forward(z[None])[0]-r,a,gtol=1e-12).x for a,r in zip(g,x)])


def branch(panel,x):
    u,v=[np.asarray(panel[k],float) for k in ('u','v')]
    def basis(u,k):
        return np.c_[affine(u),np.tanh(k[0]*u),np.sin(k[1]*u)]
    def residual(k):
        a=basis(u,k); return (a@fit(a,v)-v).ravel()
    best=min((least_squares(residual,h,bounds=(.2,6),max_nfev=100) for h in ([1,1],[2,2],[3,1.25],[1.25,3],[4,4])),key=lambda r:r.cost)
    return basis(x,best.x)@fit(basis(u,best.x),v)


def features(study):
    n=len(study['train']['x']); x=np.r_[study['train']['x'],study['test']['x']]
    z=calibrate(study['panel_a'],x); b=branch(study['panel_b'],z)
    return z,b,n


def kernelparts(z,b,n,family=0):
    def std(w):
        return (w-w[:n].mean(0))/np.maximum(w[:n].std(0),1e-9)
    joint=np.c_[z,b]; z,b,joint=std(z),std(b),std(joint)
    G=joint@joint.T/8
    Q=std(np.column_stack([joint[:,i]*joint[:,j] for i in range(8) for j in range(i,8)])); P=Q@Q.T/36
    Dz=(z[:,None,:]-z[None,:,:])**2; Db=(b[:,None,:]-b[None,:,:])**2
    if family==0:
        return np.array([np.ones_like(G),G,P,*[np.exp(-(Dz.mean(2)+Db.mean(2))/2/l) for l in (1,4,12)]])
    return np.array([np.ones_like(G),G,P,*[np.exp(-w/l) for w in (Dz.mean(2),Db.mean(2),(Dz.mean(2)+Db.mean(2))/2) for l in (1,4,12)], *[np.exp(-d/l).mean(2) for d in (Dz,Db) for l in (1,4,12)]])


def solve_batch(study):
    z,b,n=features(study); y=study['train']['y']; t=study['train']['traj']; K=kernelparts(z,b,n,1)
    B,f=sine_branch(study['panel_b'],z)
    yh=predict_readout(z,b,B,f,n,y)
    G=K[0]*20+K[1]+K[2]; state=G[n:,:n]@cho_solve(cho_factor(G[:n,:n]+.0003*np.eye(n)),t)
    last=state[:,-1]
    return {'y_hat':yh,'state_pred':state,'diagnostics_vec':np.array([yh.mean(),yh.std(),last.mean(),last.std()])}


def sine_branch(panel,z):
    u,v=[np.asarray(panel[k],float) for k in ('u','v')]
    def basis(x,h):return np.c_[affine(x),np.sin(h*x)]
    def res(h):
        A=basis(u,h[0]);return (A@fit(A,v)-v).ravel()
    h=min((least_squares(res,[i],bounds=(.4,6.5)) for i in (1.5,2.8,4.3,5.8)),key=lambda r:r.cost).x[0]
    return basis(z,h)@fit(basis(u,h),v),h


def readout_matrices(z,b,B,f,n):
    def norm(x):return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-8)
    a=norm(z); b=norm(b); B=norm(B); h=norm(np.sin(f*z)); c=norm(z-.4*np.sin(f*z)); H=np.c_[a,h]; Z=np.c_[a,B]; W=np.c_[a,b]; C=np.c_[a,c]; one=np.ones((len(a),len(a)))
    ans={}
    def add(key,value):
        if key in selected_readout_keys: ans[key]=value
    for j,X in enumerate((Z,W,H,C,B,a)):
        p=X.shape[1]; G=X@X.T/p; G1=one+G; d=((X[:,None,:]-X[None,:,:])**2).mean(2)
        for degree in (1,2,3):
            for l in (2.,6.,16.,35.):
                for mix in (.6,3.,15.): add((j,degree,l,mix),G1**degree+mix*np.exp(-d/l))
    for j,D in enumerate((h,c,B,b)):
        mixed=((a[:,None,:]-a[None,:,:])**2+(D[:,None,:]-D[None,:,:])**2)*.5
        for l in (.3,1.,3.,8.):
            k=np.exp(-mixed/l); p=np.mean(k,axis=2); p2=(k.sum(2)**2-(k*k).sum(2))/12; p3=(k.sum(2)**3-3*k.sum(2)*(k*k).sum(2)+2*(k**3).sum(2))/24
            G=one+(a@a.T+D@D.T)/8
            for m in (.1,1.,10.):
                add((6+j,1,l,m),G+m*(p+.2*p2+.02*p3))
                add((6+j,2,l,m),G**2+m*(p+p2+.2*p3))
    return ans


READOUT_MODELS = {((7, 2, 8.0, 1.0), 0.045): 1.0}
selected_readout_keys = {k for k,r in READOUT_MODELS}

def predict_readout(z,b,B,f,n,y):
    matrices=readout_matrices(z,b,B,f,n); m=y.mean(); pred=np.zeros(len(z)-n); I=np.eye(n)
    for (k,r),w in READOUT_MODELS.items():
        A=matrices[k]; L=cho_factor(A[:n,:n]+r*I); pred+=w*(A[n:,:n]@cho_solve(L,y-m))
    return m+pred
