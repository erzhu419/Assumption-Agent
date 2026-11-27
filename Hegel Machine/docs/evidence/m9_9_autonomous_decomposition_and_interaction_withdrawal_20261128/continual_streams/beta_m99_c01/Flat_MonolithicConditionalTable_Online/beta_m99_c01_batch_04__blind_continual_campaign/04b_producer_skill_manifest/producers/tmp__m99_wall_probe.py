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

def axis(a,kind):
    k,v,h,r=(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1),s['robin_'+a]); return {'R':lambda:robin._axis(k,v,h,r),'S':lambda:sealed._axis(k,v,h),'P':lambda:periodic._axis(k,v,h)}[kind]()
nx,ny=s['nx'],s['ny']; U=s['source_modes']; ii,jj=s['sensor_ij'].T; idx=ii+nx*jj; Z=s['Z_nuisance']; rr=s['nuisance_ridge']; sr=s['source_ridge']; n=len(U); ns=len(idx)
def forward(a,b):
    L=(kron(eye(ny),axis('x',a))+kron(axis('y',b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); rhs=U.reshape(n,-1,order='F').T.copy(); return np.concatenate([splu(L+mu*eye(nx*ny,format='csc')).solve(rhs)[idx] for mu in s['mu_shifts']])
def fit(S,y,keep):
    z=Z[keep]; P=np.linalg.solve(z.T@z+rr*np.eye(Z.shape[1]),z.T); A=S[keep]-z@(P@S[keep]); b=y[keep]-z@(P@y[keep]); w,nu=robin._box_simplex(A.T@A+sr*np.eye(n),A.T@b,s['u_box'],s['total_budget']); gamma=P@(y[keep]-S[keep]@w); return w,y-S@w-Z@gamma
from scipy.optimize import least_squares
Y=robin._calibrate(s)
I=eye(nx*ny,format='csc'); rhs=U.reshape(n,-1,order='F').T.copy(); V=diags(s['V_reaction_2d'].ravel(order='F'))
xx=[axis('x',k) for k in 'SRP']; yy=[axis('y',k) for k in 'SRP']
def model(p):
    ax=xx[0]+p[0]*(xx[1]-xx[0])+p[1]*(xx[2]-xx[0]); ay=yy[0]+p[2]*(yy[1]-yy[0])+p[3]*(yy[2]-yy[0]); L=(kron(eye(ny),ax)+kron(ay,eye(nx))+V).tocsc()
    return np.concatenate([splu(L+mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
def project_fit(S,keep):
    # Diagnose model error separately from source shrinkage, with equality enforced.
    D=np.column_stack((S[:,:-1]-S[:,[-1]], Z)); q=Y-s['total_budget']*S[:,-1]; b=np.linalg.lstsq(D[keep],q[keep],rcond=None)[0]; w=np.r_[b[:n-1],s['total_budget']-sum(b[:n-1])]; return q-D@b,w
keep=np.ones(len(Y),bool)
for a,b in product(range(3),repeat=2):
    S=forward('SRP'[a],'SRP'[b]); e,w=project_fit(S,keep); print('SRP'[a]+'SRP'[b], 'unregularized RMS %.6g'%np.sqrt(np.mean(e**2)), 'range',np.round([w.min(),w.max()],4))
for mask in [keep]+[np.arange(len(Y))//ns!=k for k in range(len(s['mu_shifts']))]:
    res=least_squares(lambda p:project_fit(model(p),mask)[0][mask],[0,0,0,1],bounds=(0,2),ftol=1e-10,gtol=1e-11); e,w=project_fit(model(res.x),mask); print('mix',res.x.round(5),'train RMS',np.sqrt(np.mean(e[mask]**2)),'test RMS',np.sqrt(np.mean(e[~mask]**2)) if not mask.all() else None,'w',w.round(4))
e,w=project_fit(forward('S','P'),keep); print('SP per-shift RMS',np.sqrt(np.mean(e.reshape(-1,ns)**2,axis=1)), 'largest',[(int(j),round(e[j],5),idx[j%ns]) for j in np.argsort(abs(e))[-6:]])
print('calibration inversion span',Y.min(),Y.max())
