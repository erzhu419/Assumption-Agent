import pickle
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = item['study']; pulses = item['campaign_diagnostics']
c = s['sensor_calibration_sweeps']; ref = c['ref_rosette']; raw = c['raw_rosette']
B = np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],raw,rcond=None)[0]
y = np.stack([np.linalg.solve(B[1:].T,(p['y_pulse_raw'].reshape(-1,4)-B[0]).T).T.ravel() for p in pulses],axis=1)
nx,ny=s['nx'],s['ny']; rows=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; eye=sparse.eye(nx*ny,format='csc')
W=np.column_stack([p['w_pulse'] for p in pulses]); rhs=np.column_stack([a.ravel(order='F') for a in s['source_modes']])@W
Z=s['Z_nuisance']; Q=np.linalg.qr(Z,mode='reduced')[0]
def axis(k,v,h,r,leak,fac):
    face=(k[1:]+k[:-1])/(2*h*h)
    left=np.r_[-face[:-1]-np.maximum(v[1:-1],0)/h,-fac[1]*k[-1]/h**2]
    right=np.r_[-fac[0]*k[0]/h**2,-face[1:]-np.maximum(-v[1:-1],0)/h]
    return sparse.diags((left,np.r_[-right[0]+leak[0]*r/h,-left[:-1]-right[1:],-left[-1]+leak[1]*r/h],right),(-1,0,1),format='csc')
def pred(t):
    ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],t[:2],t[4:6]); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],t[2:4],t[6:8])
    A=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    return np.vstack([splu(A+mu*eye).solve(rhs)[rows] for mu in s['mu_shifts']])
def fmt(e): return np.round(np.sqrt(np.mean(e**2,axis=0)),8)
for t in [np.r_[np.ones(8)],np.r_[np.zeros(4),np.ones(4)]]:
    e=pred(t)-y; print('baseline walls',t[:4],'RMSE',fmt(e),'after nuisance',fmt(e-Q@(Q.T@e)))
for model in ['xy','four','eight']:
    def expand(p):
        if model=='xy': return np.r_[np.repeat(p,2),np.ones(4)]
        if model=='four': return np.r_[p,np.ones(4)]
        return p
    n={'xy':2,'four':4,'eight':8}[model]
    for profile in [False,True]:
        def errors(p):
            e=pred(expand(p))-y
            return e-Q@(Q.T@e) if profile else e
        for train in [[0,1],[0],[1]]:
            fit=least_squares(lambda p:errors(p)[:,train].ravel(),np.ones(n),bounds=(0,np.inf),gtol=1e-12,ftol=1e-12,xtol=1e-12)
            e=pred(expand(fit.x))-y
            ee=errors(fit.x)
            print(model,profile,train,'walls',np.round(expand(fit.x),6),'raw',fmt(e),'fit metric',fmt(ee),'worst shift/pulse',np.max(np.sqrt(np.mean(ee.reshape(3,-1,2)**2,axis=1))))
