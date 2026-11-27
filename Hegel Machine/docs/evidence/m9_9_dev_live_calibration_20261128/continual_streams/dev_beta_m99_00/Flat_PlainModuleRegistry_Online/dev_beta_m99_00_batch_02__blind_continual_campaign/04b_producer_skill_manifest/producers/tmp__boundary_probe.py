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
        axes.append(diags((lo,d,up),(-1,0,1),format='csc'))
    L=(kron(eye(ny),axes[0])+kron(axes[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); rhs=modes.transpose(0,2,1).reshape(-1,nx*ny).T
    return np.vstack([splu(L+mu*I).solve(rhs)[pos] for mu in s['mu_shifts']])

if __name__=='__main__':
    item=pickle.load(open('public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('_upstream_param_payload.json'))
    pulses=item['campaign_diagnostics']; w=np.stack([p['w_pulse'] for p in pulses]); modes=np.einsum('pm,mxy->pxy',w,s['source_modes']); print('weights of pulses',w)
    y=np.stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses],axis=1)
    Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; print('pulse amplitude rms',np.sqrt(np.mean(y*y,axis=0)))
    def errs(p): return forward(s,p,modes)-y
    def report(name,p):
        e=errs(p); q=e-Q@(Q.T@e)
        print(name,np.round(p,6),'raw',np.sqrt(np.mean(e*e,axis=0)).round(8),'projected',np.sqrt(np.mean(q*q,axis=0)).round(8))
    for name,robs in [('open',[1]*4),('sealed',[0]*4),('sealed x',[0,0,1,1]),('sealed y',[1,1,0,0])]: report(name,np.r_[robs,[1]*4])
    for projected in (False,True):
        def residual(z, fitdiff):
            e=errs(z if fitdiff else np.r_[z,[1]*4]); return (e-Q@(Q.T@e) if projected else e).ravel()
        for fitdiff in (False,True):
            p=least_squares(residual,np.ones(8 if fitdiff else 4),bounds=(0,10),args=(fitdiff,),gtol=1e-13,ftol=1e-13,xtol=1e-13).x
            report(('projected' if projected else 'raw')+(' full' if fitdiff else ' leakage'),p if fitdiff else np.r_[p,[1]*4])
