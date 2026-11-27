import pickle, runpy, json
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import splu
p=runpy.run_path('/tmp/hegel_continual_archive/beta_m99_c04_s00_solver.py')
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; nx,ny=s['nx'],s['ny']; pulses=a['campaign_diagnostics']; wp=np.array([v['w_pulse'] for v in pulses]).T; y=np.array([p['_calibrate'](dict(s,y_obs_raw=v['y_pulse_raw'])) for v in pulses]).T
ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; modes=s['source_modes']; rhs=np.stack([m.ravel(order='F') for m in modes],axis=1)@wp
V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc')
def pred(theta):
    axes=[]
    for ax,n,pars in zip('xy',(nx,ny),theta.reshape(2,2,3)):
        h=s['L'+ax]/(n-1); k=s['kappa_'+ax]; rob=s['robin_'+ax]
        A=p['_axis'](k,s['vel_'+ax],h,rob).tolil()
        for i,j,l,(r,d,wrap) in zip((0,n-1),(1,n-2),(n-1,0),pars):
            A[i,i]=r*rob/h+(d+wrap)*k[i]/h**2; A[i,j]=-d*k[i]/h**2; A[i,l]=-wrap*k[i]/h**2
        axes.append(A.tocsc())
    op=kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')+V
    return np.vstack([splu(op+mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
def report(name,theta):
    e=(pred(np.asarray(theta))-y).reshape(3,len(idx),len(pulses)); print(name,'rmse',np.sqrt(np.mean(e**2)), 'shift-pulse', np.sqrt(np.mean(e**2,axis=1)).round(8).tolist())
base=np.tile([1.,1.,0.],4)
for name,theta in [('open',base),('sealed',np.tile([0,1,0.],4)),('reflected',np.tile([0,2,0.],4)),('periodic',np.tile([0,1,1.],4))]: report(name,theta)
for name,cols in [('Robin per wall',[0,3,6,9]),('Robin and diffusion',[0,1,3,4,6,7,9,10]),('Robin diffusion wrap',list(range(12)))]:
    def fill(z):
        t=base.copy(); t[cols]=z; return t
    for omit in [-1,0,1,2]:
        ids=np.arange(3)!=omit
        fit=least_squares(lambda z:(pred(fill(z))-y)[:,ids].ravel(),base[cols],bounds=(-0.02,5),gtol=1e-12,xtol=1e-12,ftol=1e-12)
        print('parameters',name,'omit',omit,fill(fit.x).round(6).reshape(4,3).tolist()); report(name,fill(fit.x))
for entry in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
    t=np.array(entry['facility_telemetry_trace']); print(entry['completed_campaign_short_name'],'trace mean',t.mean(0).round(4),'std',t.std(0).round(4))
