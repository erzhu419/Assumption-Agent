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
def paxis(k,v,h):
    n=len(k); i=np.arange(n); prev=(i-1)%n; nxt=(i+1)%n
    left=-(k+k[prev])/(2*h**2)-np.maximum(v,0)/h
    right=-(k+k[nxt])/(2*h**2)-np.maximum(-v,0)/h
    return sparse.csc_matrix((np.r_[left,-left-right,right],(np.tile(i,3),np.r_[prev,i,nxt])),shape=(n,n))
ax=paxis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1)); ay=paxis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1))
for a,k,L,n in [(ax,s['kappa_x'],s['Lx'],nx),(ay,s['kappa_y'],s['Ly'],ny)]:
    h=L/(n-1); print('inward',[-a[0,1]*h*h/k[0],-a[-1,-2]*h*h/k[-1]],'wrap',[-a[0,-1]*h*h/k[0],-a[-1,0]*h*h/k[-1]],'conservation',np.max(np.abs(a@np.ones(n))))
A=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
e=np.vstack([splu(A+mu*eye).solve(rhs)[rows] for mu in s['mu_shifts']])-y
print('periodic pulse',fmt(e),'shift by pulse',np.sqrt(np.mean(e.reshape(3,-1,2)**2,axis=1)))
