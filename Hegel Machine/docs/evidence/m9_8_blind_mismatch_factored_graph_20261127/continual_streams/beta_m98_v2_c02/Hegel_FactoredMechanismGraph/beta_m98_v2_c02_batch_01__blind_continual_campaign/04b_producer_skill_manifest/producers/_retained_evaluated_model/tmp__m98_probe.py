import importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/beta_m98_v2_c02_s00_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; pulses=it['campaign_diagnostics']; W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([prior._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; nx,ny=s['nx'],s['ny']; axes=[prior._axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1),s['robin_'+a]) for a in 'xy']; src=s['source_modes'].reshape(-1,nx*ny,order='F').T@W; rows=s['sensor_ij']@np.array([1,nx]); I=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def predict(p=None,kind='local'):
    out=[]
    for ai,(A,a) in enumerate(zip(axes,'xy')):
        B=A.toarray()
        if p is not None:
            for side,(i,d) in enumerate([(0,1),(len(B)-1,-1)]):
                k=p[ai,side]; B[i,i]+=k[0]; B[i,i+d]+=k[1]
                if kind=='far': B[i,len(B)-1-i]+=k[2]
                if kind=='next': B[i,i+2*d]+=k[2]
        out.append(sparse.csc_matrix(B))
    L=sparse.kron(sparse.eye(ny),out[0],format='csc')+sparse.kron(out[1],sparse.eye(nx),format='csc')+V
    return np.vstack([splu(L+mu*I).solve(src)[rows] for mu in s['mu_shifts']])
def project(r): return r-Q@(Q.T@r)
def rms(r): return np.sqrt(np.mean(r*r,axis=0))
if __name__=='__main__':
    np.set_printoptions(precision=6,suppress=True); base=predict(); print('pulse norms and baseline errors',rms(Y), rms(base-Y),rms(project(base-Y))); print('baseline nuisance coefficients',np.linalg.lstsq(Z,Y-base,rcond=None)[0]); print('axis dx kap vel rob');
    for a in 'xy': print(a,s['L'+a]/(s['n'+a]-1),s['kappa_'+a][[0,1,-2,-1]],s['vel_'+a][[0,-1]],s['robin_'+a])
    print('observation scale',np.min(Y),np.max(Y)); print('pulse weights',W.T)
    for kind,n in [('local',2),('far',3),('next',3)]:
        for cols in [[0,1],[0],[1]]:
            def f(v): return project(predict(v.reshape(2,2,n),kind)-Y)[:,cols].ravel()
            r=least_squares(f,np.zeros(4*n),max_nfev=120,ftol=1e-11,gtol=1e-12,xtol=1e-11)
            k=r.x.reshape(2,2,n); delta=predict(k,kind)-Y
            print(kind,cols,'raw',rms(delta),'projected',rms(project(delta)),'params',k.reshape(4,n).tolist())
