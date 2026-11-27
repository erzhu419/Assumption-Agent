import itertools,pickle
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares


def axis(k,v,L,r,grammar):
    spacing,avg,adv,edge=grammar
    n=len(k); h=L/(n+spacing)
    d={'ar':(k[:-1]+k[1:])/2,'ha':2*k[:-1]*k[1:]/(k[:-1]+k[1:]),'ge':np.sqrt(k[:-1]*k[1:])}[avg]/h**2
    vf=(v[:-1]+v[1:])/2/h
    p=np.maximum(vf,0); m=np.maximum(-vf,0)
    lo=-d-p; up=-d-m
    diag=np.zeros(n)
    diag[:-1]+=d; diag[1:]+=d
    if adv=='flux': diag[:-1]+=p; diag[1:]+=m
    elif adv=='rows': diag[:-1]+=m; diag[1:]+=p
    elif adv=='cell':
        lo=-d-np.maximum(v[1:],0)/h; up=-d-np.maximum(-v[:-1],0)/h
        diag+=np.abs(v)/h
    elif adv=='cellclamp':
        lo=-d-np.maximum(v[1:],0)/h; up=-d-np.maximum(-v[:-1],0)/h
        diag[:-1]+=np.maximum(-v[:-1],0)/h; diag[1:]+=np.maximum(v[1:],0)/h
    if edge=='out': diag[[0,-1]]+=np.maximum([-v[0],v[-1]],0)/h
    if edge=='abs': diag[[0,-1]]+=np.abs(v[[0,-1]])/h
    if edge=='in': diag[[0,-1]]+=np.maximum([v[0],-v[-1]],0)/h
    diag[[0,-1]]+=np.broadcast_to(r,(2,))
    return sp.diags((lo,diag,up),(-1,0,1),format='csc')


def predict(s,r,grammar,w=None,last=False):
    nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T
    A=sp.kron(sp.eye(ny),axis(s['kappa_x'],s['vel_x'],s['Lx'],r[:2],grammar))+sp.kron(axis(s['kappa_y'],s['vel_y'],s['Ly'],r[2:],grammar),sp.eye(nx))+sp.diags(s['V_reaction_2d'].ravel(order='F'))
    F=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
    if w is not None: F=F@w
    I=sp.eye(nx*ny,format='csc'); obs=[]
    for mu in s['mu_shifts']:
        U=splu((A+mu*I).tocsc()).solve(F)
        obs.append(U[i+nx*j])
    return (np.vstack(obs),U) if last else np.vstack(obs)

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; print('calibration identity RMS',np.std(y-x),'linear',np.std(y-np.c_[np.ones(len(x)),x]@np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0])); print('telemetry summary',np.mean(s['aux_manifold_telemetry'],axis=0),np.std(s['aux_manifold_telemetry'],axis=0)); print('kappa endpoints',s['kappa_x'][[0,-1]],s['kappa_y'][[0,-1]])
 W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; P=lambda a:a-Q@(Q.T@a)
 scores=[]
 for g in itertools.product((-1,0,1),('ar','ha','ge'),('flux','rows','cell','cellclamp'),('none','out','in','abs')):
    def fun(r): return P(predict(s,r,g,W)-Y).ravel()
    fit=least_squares(fun,[3]*4,bounds=(-3.,70.),max_nfev=25)
    scores.append((np.sqrt(np.mean(fit.fun**2)),g,fit.x,np.max(np.abs(fit.fun))))
 for a in sorted(scores,key=lambda a:a[0])[:18]: print(a)
