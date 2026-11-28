import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
from scipy.sparse import eye, kron, diags
from scipy.sparse.linalg import splu
p='/tmp/hegel_continual_archive/beta_m100_c01_s00_solver.py'
sp=importlib.util.spec_from_file_location('old',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; nx,ny=s['nx'],s['ny']; pd=i['campaign_diagnostics']; ws=np.array([p['w_pulse'] for p in pd]); c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; b=np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],c['raw_rosette'],rcond=None)[0]; y=np.column_stack([np.linalg.solve(b[1:].T,(p['y_pulse_raw'].reshape(-1,4)-b[0]).T).T.ravel() for p in pd]); ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]; rhs=np.column_stack([x.ravel(order='F') for x in (ws@s['source_modes'].reshape(ws.shape[1],-1)).reshape(-1,nx,ny)]); E=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; hs=np.array([s['Lx']/(nx-1),s['Ly']/(ny-1)]); base=[m._axis(s['kappa_'+a],s['vel_'+a],h,s['robin_'+a]).toarray() for a,h in zip('xy',hs)]
def predict(q):
    axes=[]
    for A,h,a,ps in zip(base,hs,'xy',q.reshape(2,2,3)):
        A=A.copy(); kap=s['kappa_'+a]; rob=s['robin_'+a]; n=len(A)
        for j,k,other,(f,r,t) in zip((0,n-1),(1,n-2),(n-1,0),ps):
            A[j,k]=-f*kap[j]/h**2; A[j,other]=-t*kap[j]/h**2; A[j,j]=-A[j,k]-A[j,other]+r*rob/h
        axes.append(A)
    A=kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')+V
    return np.vstack([splu(A+mu*E).solve(rhs)[slots] for mu in s['mu_shifts']])
def report(e):
    return np.round(np.sqrt(np.mean(e**2,axis=0)),9)
for name,q in [('open',[1,1,0]*4),('reflect',[1,0,0]*4),('ghost-reflect',[2,0,0]*4),('periodic',[1,0,1]*4),('x-sealed',[1,0,0]*2+[1,1,0]*2),('y-sealed',[1,1,0]*2+[1,0,0]*2)]:
    e=predict(np.array(q,dtype=float))-y; print(name,report(e),report(e-Q@(Q.T@e)))
for nuisance in (False,True):
    for omit in (-1,0,1,2):
        sel=np.arange(y.shape[1])!=omit
        def residual(q):
            e=predict(q)-y
            if nuisance: e=e-Q@(Q.T@e)
            return e
        sol=least_squares(lambda q:residual(q)[:,sel].ravel(),[1,.2,0]*4,bounds=(np.tile([0,-.2,-.01],4),np.tile([8,5,4],4)),max_nfev=90,gtol=1e-12)
        print('fit nuisance',nuisance,'omit',omit,'coefficients',np.round(sol.x.reshape(4,3),6).tolist(),'pulse rms',report(residual(sol.x)))
print('aux mean std',np.mean(s['aux_manifold_telemetry'],axis=0),np.std(s['aux_manifold_telemetry'],axis=0))
