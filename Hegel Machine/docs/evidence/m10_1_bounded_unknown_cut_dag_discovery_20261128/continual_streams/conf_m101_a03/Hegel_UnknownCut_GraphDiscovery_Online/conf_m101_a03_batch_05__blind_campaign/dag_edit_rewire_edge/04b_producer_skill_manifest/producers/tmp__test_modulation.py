exec(open('/tmp/sparse_readout.py').read().split('for base in')[0])
from scipy.optimize import least_squares
folds=folds+[np.argsort(np.linalg.norm(x,axis=1))[-14:]]
u=x[:,1]-x[:,3]; p=z[:,0]*z[:,2]; q=z[:,1]*z[:,3]; B=np.c_[np.ones(n),z,p,q]
def forms(f,g):
    a=x[:,1]; d=x[:,3]; U=np.sin(f*(a-g*d)); return U
mods={'none':np.empty((n,0)), 'cubic':(z[:,0]*x[:,1]**2)[:,None], 'quotient':(z[:,0]/(1+.5*x[:,1]**2))[:,None], 'quadrature':np.c_[z[:,0]*x[:,1]**2,z[:,0]*x[:,3]**2,z[:,0]*x[:,1]*x[:,3]], 'pairx':np.c_[z[:,0]*x[:,1]**2,p*x[:,1]], 'terminal':T[:,[-1]], 'operator_cubic':(z[:,0]*z[:,2]**2)[:,None]}
for nm,M in mods.items():
 for trail in [False,True] if nm=='cubic' else [False]:
    BB=np.c_[B,M,T[:,-1]] if trail else np.c_[B,M]
    E=[]; W=[]
    for v in folds+[np.array([],int)]:
        t=np.setdiff1d(idx,v)
        def run(par,ret=False):
            ff=np.c_[BB,forms(*par)]; w=np.linalg.lstsq(ff[t],y[t],rcond=None)[0]; r=y-ff@w
            return (ff,r,w) if ret else r[t]
        par=min((least_squares(run,[f,1],bounds=([.4,.2],[5,2]),max_nfev=70) for f in [1.3,2.4,3.6]),key=lambda g:g.cost).x
        A,r,w=run(par,True); E.append([np.abs(r[v]).mean() if len(v) else 0,np.std(r[t])]); W.append(np.r_[par,w]);
    print(nm,trail,np.round(E,4).T.tolist(),'coef full',np.round(W[-1],3));
    if nm=='cubic' and not trail:
        H=F-A@np.linalg.lstsq(A,F,rcond=None)[0]; H/=np.maximum(np.linalg.norm(H,axis=0),1e-7); jj=np.argsort(-np.abs(r@H))[:18]; print('remaining',list(zip(N[jj],np.round((r@H)[jj],4)))); print('large residuals',np.round(np.c_[x,z,y,r][np.argsort(abs(r))[-8:]],3))
# Does the scalar correction share the same amplitude limit as the trajectory?
Q=np.c_[B,forms(1.7,1),mods['cubic']]; R=Y-Q@np.linalg.lstsq(Q,Y,rcond=None)[0]; print('residual correlation with trajectory',np.corrcoef(R.T)[0]); print('terminal regression',np.linalg.lstsq(B[:,:5],T[:,-1],rcond=None)[0]);
