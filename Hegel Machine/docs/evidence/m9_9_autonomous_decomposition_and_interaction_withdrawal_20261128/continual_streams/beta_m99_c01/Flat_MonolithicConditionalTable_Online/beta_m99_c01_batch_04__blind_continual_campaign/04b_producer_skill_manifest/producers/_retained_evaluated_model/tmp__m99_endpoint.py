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
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
full=np.ones(len(Y),bool)
shifts=np.arange(len(Y))//ns
box=s['u_box']; budget=s['total_budget']
def ends(a,p):
    k=s['kappa_'+a]; h=s['L'+a]/(s['n'+a]-1); m=len(k); r=s['robin_'+a]; B=axis(a,'S').tolil()
    for j,i,inward,wrap in [(0,0,1,m-1),(1,m-1,m-2,0)]:
        own=k[i]/h**2; face=(k[i]+k[inward])/(2*h**2); seam=(k[i]+k[wrap])/(2*h**2); vel=s['vel_'+a][i]*(-1 if j else 1)/h
        local=own+p[0]*(face-own+max(-vel,0)); cross=p[1]*(seam+max(vel,0)); leak=p[2]*r/h
        B[i,i]=local+cross+leak; B[i,inward]=-local; B[i,wrap]=-cross
    return B.tocsc()
def expanded(p):
    L=(kron(eye(ny),ends('x',p[:3]))+kron(ends('y',p[3:]),eye(nx))+V).tocsc()
    return np.concatenate([splu(L+mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
def phys(S,mask):
    z=Z[mask]; Q=np.linalg.qr(z,mode='reduced')[0]; D=S[mask]-Q@(Q.T@S[mask]); y=Y[mask]-Q@(Q.T@Y[mask]);
    def f(w):
        e=D@w-y; return .5*(e@e),D.T@e
    w=minimize(f,np.full(n,budget/n),jac=True,bounds=[(0,box)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(n)}],method='SLSQP',options={'ftol':1e-13}).x
    gamma=np.linalg.lstsq(z,Y[mask]-S[mask]@w,rcond=None)[0]
    return Y-S@w-Z@gamma,w
for tag,S in [(a+b,forward(a,b)) for a,b in product('SP',repeat=2)]:
    e,w=phys(S,full); test=[]
    for k in np.unique(shifts):
        off=shifts==k; es,_=phys(S,~off); test.append(np.sqrt(np.mean(es[off]**2)))
    print(tag,'physical fit',np.sqrt(np.mean(e**2)),'held shifts',np.round(test,6),'w',w.round(3))
for mask in [full]+[shifts!=k for k in np.unique(shifts)]:
    opt=least_squares(lambda p:phys(expanded(p),mask)[0][mask],[.5,.1,.001,.5,.8,.001],bounds=([-4,0,0]*2,[4,3,2]*2),diff_step=1e-4,ftol=1e-8,max_nfev=100)
    e,w=phys(expanded(opt.x),mask); print('endpoint',opt.x.round(5),'train',np.sqrt(np.mean(e[mask]**2)),'held',np.sqrt(np.mean(e[~mask]**2)) if not mask.all() else None,'w',w.round(3))
