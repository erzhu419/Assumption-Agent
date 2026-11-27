import json,pickle,sys
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def forward(s, p, modes):
    nx,ny=s['nx'],s['ny']; axes=[]
    for j,(key,n) in enumerate(zip('xy',(nx,ny))):
        h=s['L'+key]/(n-1); k=s['kappa_'+key]; v=s['vel_'+key]; face=(k[:-1]+k[1:])/(2*h*h)
        lo=np.r_[-face[:-1]-np.maximum(v[1:-1],0)/h,-p[5+2*j]*k[-1]/h**2]
        up=np.r_[-p[4+2*j]*k[0]/h**2,-face[1:]-np.maximum(-v[1:-1],0)/h]
        d=np.r_[-up[0]+p[2*j]*s['robin_'+key]/h,-lo[:-1]-up[1:],-lo[-1]+p[2*j+1]*s['robin_'+key]/h]
        a=diags((lo,d,up),(-1,0,1),format='lil')
        if len(p)>8:
            wrap=np.array([p[8+2*j],p[9+2*j]])*k[[0,-1]]/h**2
            a[0,-1]=-wrap[0]; a[-1,0]=-wrap[1]
            a[0,0]+=wrap[0]; a[-1,-1]+=wrap[1]
        axes.append(a.tocsc())
    L=(kron(eye(ny),axes[0])+kron(axes[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); rhs=modes.transpose(0,2,1).reshape(-1,nx*ny).T
    return np.vstack([splu(L+mu*I).solve(rhs)[pos] for mu in s['mu_shifts']])

from itertools import product
item=pickle.load(open('public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('_upstream_param_payload.json')); pulses=item['campaign_diagnostics']
modes=np.stack([np.tensordot(p['w_pulse'],s['source_modes'],axes=1) for p in pulses]); y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses]);
print('sweep raw affine coefficients',np.linalg.lstsq(np.c_[np.ones(len(s['sensor_calibration_sweeps']['ref_rosette'])),s['sensor_calibration_sweeps']['ref_rosette']],s['sensor_calibration_sweeps']['raw_rosette'],rcond=None)[0].round(6)); print('local telemetry median',np.median(s['aux_manifold_telemetry'],axis=0))
def res(p): return forward(s,p,modes)-y
opts={'open':[1,1,1,1,0,0],'sealed':[0,0,1,1,0,0],'wrap':[0,0,1,1,1,1],'link':[0,0,0,0,1,1],'mirror':[0,0,2,2,0,0]}
rows=[]
for (a,x),(b,z) in product(opts.items(),repeat=2):
 p=np.array([x[:2],z[:2],x[2:4],z[2:4],x[4:],z[4:]]).ravel(); e=res(p); rows.append((np.mean(e*e)**.5,a,b,p))
for err,a,b,p in sorted(rows,key=lambda x:x[0])[:8]: print(a,b,err,np.mean(res(p)**2,axis=0)**.5)
for a,b in [('wrap','wrap'),('wrap','sealed'),('sealed','wrap')]:
 p=next(p for _,aa,bb,p in rows if (a,b)==(aa,bb)); fit=least_squares(lambda v:res(v).ravel(),p,bounds=(0,20),gtol=1e-13,ftol=1e-13,xtol=1e-13);print('fit',a,b,fit.x.round(7),np.mean(res(fit.x)**2,axis=0)**.5)
