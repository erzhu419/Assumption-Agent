import sys
sys.path.insert(0,'/tmp')
from m98_probe import load,root,st,ps,X,Y,Z
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import eye,diags,kron
from scipy.sparse.linalg import splu
m=load(root+'beta_m98_v2_c04_batch_02_solver.py')
nx,ny=st['nx'],st['ny']; dims=(nx,ny)
base=[]; scales=[]
for a,n in zip('xy',dims):
    k=st['kappa_'+a]; L=st['L'+a]; h=L/(n-1); r=st['robin_'+a]; v=st['vel_'+a]
    base.append(m._axis(k,v,L,0).tolil())
    scales.append(np.array([[r/h,k[0]/h**2,k[0]/h**2],[r/h,k[-1]/h**2,k[-1]/h**2]]))
    print('axis',a,'h, robin, edge kappa, edge velocity',h,r,k[[0,1,-2,-1]],v[[0,-1]])
x=X-X.mean(0); gain=np.sum(x*(Y-Y.mean(0)),axis=0)/np.sum(x*x,axis=0); offset=Y.mean(0)-gain*X.mean(0)
target=np.column_stack([((p['y_pulse_raw'].reshape(-1,4)-offset)/gain).ravel() for p in ps]); weights=np.stack([p['w_pulse'] for p in ps],axis=1);print('pulse weights',weights.T)
rhs=st['source_modes'].transpose(0,2,1).reshape(len(weights),nx*ny).T@weights
i,j=st['sensor_ij'].T; idx=i+nx*j
print('sensors',st['sensor_ij'].T)
I=eye(nx*ny,format='csc'); V=diags(st['V_reaction_2d'].ravel(order='F'),format='csc')
def predict(p):
    axes=[]
    for n,b,q,s in zip(dims,base,p.reshape(2,2,3),scales):
        A=b.copy()
        for (row,adj,opp),(leak,near,wrap) in zip(((0,1,n-1),(n-1,n-2,0)),q*s):
            A[row,row]=leak+near+wrap; A[row,adj]=-near; A[row,opp]=-wrap
        axes.append(A.tocsc())
    op=kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')+V
    return np.vstack([splu(op+mu*I).solve(rhs)[idx] for mu in st['mu_shifts']])
start=np.tile([0.,1.,0.],4)
for name,active in [('leak',[0]),('leak+near',[0,1]),('leak+wrap',[0,2]),('all',[0,1,2])]:
    sel=np.array([i*3+j for i in range(4) for j in active]); print('MODEL',name)
    for train in ([0],[1],[0,1]):
        def expand(x):
            p=start.copy();p[sel]=x;return p
        def residual(x):return (predict(expand(x))-target)[:,train].ravel()
        fit=least_squares(residual,start[sel],max_nfev=180,gtol=1e-12,ftol=1e-12,xtol=1e-12)
        p=expand(fit.x); e=predict(p)-target
        print(train,'rms',np.sqrt(np.mean(e*e,axis=0)),'max',np.max(np.abs(e),axis=0),'params',p.reshape(4,3).round(6).tolist(),'singular min',np.linalg.svd(fit.jac,compute_uv=False)[-1])
