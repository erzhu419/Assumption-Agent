import pickle, numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
import runpy
base = runpy.run_path('/tmp/hegel_continual_archive/dev_beta_m98_00_batch_02_solver.py')
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; ps=item['campaign_diagnostics']
cal=s['sensor_calibration_sweeps']; ref=cal['ref_rosette']; raw=cal['raw_rosette']; A=np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],raw,rcond=None)[0]
def calibrate(raw): return np.linalg.solve(A[1:].T,(raw.reshape(-1,4)-A[0]).T).T.ravel()
Y=np.column_stack([calibrate(p['y_pulse_raw']) for p in ps]); W=np.stack([p['w_pulse'] for p in ps],axis=1)
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; rhs=np.array([m.ravel(order='F') for m in s['source_modes']]).T
I=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def design(p, source=rhs, fields=False):
    axes=[]
    for j,(a,n) in enumerate(zip('xy',(nx,ny))):
        h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a]; r=s['robin_'+a]; q=p[6*j:6*j+6]
        M=base['_axis'](k,v,h,0).tolil()
        for e,i,b in ((0,0,1),(1,n-1,n-2)):
            M[i,b]=-q[e]*k[i]/h**2; M[i,n-1-i]=-q[4+e]*k[i]/h**2; M[i,i]=-M[i,b]-M[i,n-1-i]+q[2+e]*r/h
        axes.append(M.tocsc())
    ax,ay=axes; L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+V
    U=[splu(L+mu*I).solve(source) for mu in s['mu_shifts']]
    D=np.vstack([u[rows] for u in U]); return (D,U) if fields else D

def fit(convert, p0):
    def err(p): return (design(convert(p),rhs@W)-Y).reshape(3,-1,2)
    fits=[]
    masks=[np.ones((3,len(rows),2),bool)]; masks += [masks[0].copy() for _ in range(5)]
    for k in range(2): masks[1+k][:,:,k]=False
    for k in range(3): masks[3+k][k]=False
    for t,mask in enumerate(masks):
        f=least_squares(lambda p:err(p)[mask],p0,ftol=1e-12,gtol=1e-12,xtol=1e-12,max_nfev=200)
        e=err(f.x); print(t,'coeff',convert(f.x),'train',np.sqrt(np.mean(e[mask]**2)),'held',np.sqrt(np.mean(e[~mask]**2)) if np.any(~mask) else '-', 'Jac min sv',np.linalg.svd(f.jac,compute_uv=False)[-1]); fits.append(f)
    return fits
if __name__=='__main__':
    for a,b in ((0,0),(1,0),(0,1),(1,1)):
        p=[1,1,0,0,a,a,1,1,0,0,b,b]
        R=(design(p)@W-Y).reshape(3,-1,2)
        print('wrap',a,b,'RMS',np.sqrt(np.mean(R**2,axis=1)))
    fit(lambda p:p,[1,1,0,0,1,1]*2)
