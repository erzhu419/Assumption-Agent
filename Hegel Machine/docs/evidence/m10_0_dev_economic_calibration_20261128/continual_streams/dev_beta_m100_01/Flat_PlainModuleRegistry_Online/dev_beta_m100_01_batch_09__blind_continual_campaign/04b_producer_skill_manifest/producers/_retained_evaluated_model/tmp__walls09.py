from itertools import product
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize

def axis(kap,vel,h,rob,kind):
    i=np.arange(len(kap)); lo=(i-1)%len(i); hi=(i+1)%len(i)
    a=(kap+kap[lo])/(2*h*h)+np.maximum(vel,0)/h
    b=(kap+kap[hi])/(2*h*h)+np.maximum(-vel,0)/h
    leak=np.zeros(len(i))
    if kind!='P':
        a[0]=b[-1]=0; b[0]=kap[0]/h**2; a[-1]=kap[-1]/h**2
        if kind=='R': leak[[0,-1]]=rob/h
    return sparse.csc_matrix((np.r_[a+b+leak,-a,-b],(np.tile(i,3),np.r_[i,lo,hi])),shape=(len(i),)*2)

def forward(st,kinds):
    nx,ny=st['nx'],st['ny']; ix,iy=st['sensor_ij'].T
    ax,ay=[axis(st['kappa_'+j],st['vel_'+j],st['L'+j]/(st['n'+j]-1),st['robin_'+j],kind) for j,kind in zip('xy',kinds)]
    L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(st['V_reaction_2d'].ravel(order='F'),format='csc')
    rhs=st['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
    ss=[]
    for mu in st['mu_shifts']:
        u=splu(L+mu*sparse.eye(nx*ny,format='csc')).solve(rhs); ss.append(u[ix+nx*iy])
    return np.vstack(ss),u.T.reshape(-1,ny,nx).transpose(0,2,1)

def fit(st,S,train,source_ridge,nuisance_ridge):
    z=st['Z_nuisance']; y=st['y_obs_raw']; Z=z[train]; D=np.column_stack((S[train],y[train])); n=S.shape[1]; cap=st['u_box']; budget=st['total_budget']
    C=np.linalg.solve(Z.T@Z+nuisance_ridge*np.eye(Z.shape[1]),Z.T@D); B=D-Z@C; X,b=B[:,:-1],B[:,-1]; H=X.T@X+source_ridge*np.eye(n); q=X.T@b
    op=minimize(lambda w:.5*np.sum((X@w-b)**2)+.5*source_ridge*(w@w),np.full(n,budget/n),jac=lambda w:H@w-q,bounds=[(0,cap)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(n)}],method='SLSQP',options={'ftol':1e-14,'maxiter':200})
    if not op.success: raise RuntimeError(op.message)
    w=op.x
    return y-S@w-z@(C[:,-1]-C[:,:-1]@w),w

if __name__=='__main__':
    import json,pickle
    from _verified_upstream import calibrate_upstream
    st=calibrate_upstream(pickle.load(open('public_batch.pkl','rb'))[0]['study'],json.load(open('_upstream_param_payload.json')))
    y=st['y_obs_raw']; idx=np.arange(len(y)); n_sensor=len(st['sensor_ij']); rms=lambda a: np.sqrt(np.mean(np.square(a)))
    print('Each score: full RMSE, leave-one-site-rosette-out aggregate and worst RMSE, leave-one-shift-out RMSEs')
    for k in product('RSP',repeat=2):
        S,_=forward(st,k)
        for tag,sr,zr in [('contract',st['source_ridge'],st['nuisance_ridge']),('noise',0,0)]:
            r,w=fit(st,S,idx>=0,sr,zr); site=[rms(fit(st,S,~test,sr,zr)[0][test]) for t in range(n_sensor//4) for test in [idx%n_sensor//4==t]]; shift=[rms(fit(st,S,~test,sr,zr)[0][test]) for t in range(len(st['mu_shifts'])) for test in [idx//n_sensor==t]]
            print(''.join(k),tag,np.round([rms(r),rms(site),max(site),*shift],6),np.round(w,4))
