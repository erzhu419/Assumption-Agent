import itertools, importlib.util, json, pickle
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
path='/tmp/hegel_continual_archive/beta_m100_c04_batch_02_solver.py'
spec=importlib.util.spec_from_file_location('archived',path); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=a._calibrate(s); z=s['Z_nuisance']; nx,ny=s['nx'],s['ny']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']; n=len(s['source_modes']); row=s['sensor_ij']@np.array([1,nx]); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); eye=sparse.eye(nx*ny,format='csc'); react=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rob=np.repeat([s['robin_x'],s['robin_y']],2)

def forward(mult):
    leaks=np.asarray(mult)*rob
    ax=a._axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),leaks[:2]); ay=a._axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),leaks[2:]); L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+react
    return np.concatenate([splu(L+mu*eye).solve(rhs)[row] for mu in s['mu_shifts']])

def fit(S,ids):
    Z=z[ids]; sy=np.column_stack([S[ids],y[ids]]); sy-=Z@np.linalg.lstsq(Z,sy,rcond=None)[0]; A,b=sy[:,:-1],sy[:,-1]
    w,_=a._box_simplex(A.T@A+1e-12*np.eye(n),-A.T@b,cap,budget)
    gam=np.linalg.lstsq(Z,y[ids]-S[ids]@w,rcond=None)[0]
    return w,gam

def resid(mult):
    S=forward(mult); w,gam=fit(S,np.arange(len(y))); return S@w+z@gam-y

def rms(e): return np.sqrt(np.mean(e**2))

all_ids=np.arange(len(y)); groups=[all_ids.reshape(3,-1)[:,np.arange(j*4,j*4+4)].ravel() for j in range(len(row)//4)]; groups += list(all_ids.reshape(3,-1))
ret=[]
for bits in itertools.product((0,1),repeat=4):
    S=forward(bits); w,gam=fit(S,all_ids); e=S@w+z@gam-y
    cv=[]
    for test in groups:
        train=np.setdiff1d(all_ids,test); ww,gg=fit(S,train); cv.append(rms(S[test]@ww+z[test]@gg-y[test]))
    A,b=a._project(s,S,y); wr,_=a._box_simplex(A.T@A+ridge*np.eye(n),-A.T@b,cap,budget)
    ret.append((rms(e),max(cv[:8]),max(cv[8:]),bits,np.round(w,4),np.round(wr,4)))
print('fit RMSE, worst held-out rosette RMSE, worst held-out shift RMSE, walls (x-,x+,y-,y+), unregularized and production weights:')
for r in sorted(ret): print(*r)
for start in (np.full(4,.5),np.array(min(ret)[3],float)+.02):
    sol=least_squares(resid,start,bounds=(0,4),ftol=1e-11,gtol=1e-11,xtol=1e-11)
    print('continuous wall multipliers',sol.x,'rmse',rms(sol.fun),'jac singular values',np.linalg.svd(sol.jac,compute_uv=False))
