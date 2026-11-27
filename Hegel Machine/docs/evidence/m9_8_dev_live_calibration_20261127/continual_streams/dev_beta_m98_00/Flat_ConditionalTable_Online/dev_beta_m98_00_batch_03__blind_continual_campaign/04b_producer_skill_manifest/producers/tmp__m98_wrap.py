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
old_axis=axis
def axis(k,v,h,r,leak,fac,wrap):
    A=old_axis(k,v,h,r,leak,fac).tolil()
    for i,j,f in [(0,len(k)-1,wrap[0]),(len(k)-1,0,wrap[1])]:
        a=f*k[i]/h**2; A[i,i]+=a; A[i,j]-=a
    return A.tocsc()
def pred(t):
    ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],t[:2],t[4:6],t[8:10]); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],t[2:4],t[6:8],t[10:12])
    A=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    return np.vstack([splu(A+mu*eye).solve(rhs)[rows] for mu in s['mu_shifts']])
for leak in [0,1]:
 for xw,yw in [(1,0),(0,1),(1,1)]:
    t=np.r_[np.full(4,leak),np.ones(4),xw,xw,yw,yw]; print('wrap corner',leak,xw,yw,fmt(pred(t)-y))
for family in ['wrap','fac+wrap','all']:
 idx={'wrap':np.arange(8,12),'fac+wrap':np.arange(4,12),'all':np.arange(12)}[family]
 base=np.r_[np.zeros(4),np.ones(4),np.full(4,.4)]
 def expand(p):
    t=base.copy(); t[idx]=p; return t
 for training in [[0,1],[0],[1]]:
    res=least_squares(lambda p:(pred(expand(p))-y)[:,training].ravel(),base[idx],bounds=(0,8),gtol=1e-11,ftol=1e-11,xtol=1e-11)
    t=expand(res.x); e=pred(t)-y; print(family,training,np.round(t,5),fmt(e), 'worst shift',np.sqrt(np.max(np.mean(e.reshape(3,-1,2)**2,axis=1))))
print('aux mean/std',np.round(s['aux_manifold_telemetry'].mean(0),4),np.round(s['aux_manifold_telemetry'].std(0),4))
