import importlib.util, pickle, numpy as np
from itertools import product
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
p='/tmp/hegel_continual_archive/beta_m99_c01_'
def load(n):
    spec=importlib.util.spec_from_file_location(n,p+n+'_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
robin=load('batch_03'); sealed=load('batch_01'); periodic=load('batch_02'); baseline=load('s00')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; D=robin._rosette_design(x); C=robin._fit_rosette(x,y); err=y-D@C
print('rosette coeff:\n',C.round(5)); print('rosette median abs residual',np.median(abs(err),0),'large rows',np.where(np.max(abs(err),axis=1)>.005)[0]); print('ref min/max',x.min(0),x.max(0)); print('obs raw min/max',s['y_obs_raw'].reshape(-1,4).min(0),s['y_obs_raw'].reshape(-1,4).max(0))

def axis(a,kind):
    k,v,h,r=(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1),s['robin_'+a]); return {'R':lambda:robin._axis(k,v,h,r),'S':lambda:sealed._axis(k,v,h),'P':lambda:periodic._axis(k,v,h)}[kind]()
nx,ny=s['nx'],s['ny']; U=s['source_modes']; ii,jj=s['sensor_ij'].T; idx=ii+nx*jj; Z=s['Z_nuisance']; rr=s['nuisance_ridge']; sr=s['source_ridge']; n=len(U); ns=len(idx)
def forward(a,b):
    L=(kron(eye(ny),axis('x',a))+kron(axis('y',b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); rhs=U.reshape(n,-1,order='F').T.copy(); return np.concatenate([splu(L+mu*eye(nx*ny,format='csc')).solve(rhs)[idx] for mu in s['mu_shifts']])
def fit(S,y,keep):
    z=Z[keep]; P=np.linalg.solve(z.T@z+rr*np.eye(Z.shape[1]),z.T); A=S[keep]-z@(P@S[keep]); b=y[keep]-z@(P@y[keep]); w,nu=robin._box_simplex(A.T@A+sr*np.eye(n),A.T@b,s['u_box'],s['total_budget']); gamma=P@(y[keep]-S[keep]@w); return w,y-S@w-Z@gamma
for label,cal in [('full',robin),('linear',baseline),('diagonal',periodic)]:
    ycal=cal._calibrate(s); print('CAL',label)
    for a,b in product('RSP',repeat=2):
        S=forward(a,b); w,e=fit(S,ycal,np.ones(len(Z),bool)); held=[]
        for j in range(ns):
            off=np.arange(len(Z))%ns==j; _,e1=fit(S,ycal,~off); held.append(np.sqrt(np.mean(e1[off]**2)))
        print(a+b,'fit %.6g medianHeld %.6g worstHeld %.6g'%(np.sqrt(np.mean(e**2)),np.median(held),max(held)), w.round(4))
