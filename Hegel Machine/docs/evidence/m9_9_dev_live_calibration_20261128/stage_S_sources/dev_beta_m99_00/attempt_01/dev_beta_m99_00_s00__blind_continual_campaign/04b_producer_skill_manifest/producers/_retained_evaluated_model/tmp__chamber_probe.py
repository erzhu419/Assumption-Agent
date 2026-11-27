import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

def axis(kap, vel, h, rob, wall=(1,1,1,1)):
    n = len(kap)
    a = np.zeros((n,n))
    i = np.arange(1,n-1)
    km = (kap[i]+kap[i-1])/(2*h*h)
    kp = (kap[i]+kap[i+1])/(2*h*h)
    a[i,i-1] = -km-np.maximum(vel[i],0)/h
    a[i,i+1] = -kp-np.maximum(-vel[i],0)/h
    a[i,i] = km+kp+np.abs(vel[i])/h
    for j, neighbor, d, r in ((0,1,wall[0],wall[1]),(-1,-2,wall[2],wall[3])):
        a[j,neighbor] = -d*kap[j]/h**2
        a[j,j] = -a[j,neighbor]+r*rob/h
    return a

def fields(s, wall=(1,)*8):
    nx, ny = s['nx'],s['ny']
    ax,ay = (axis(s['kappa_'+k],s['vel_'+k],s['L'+k]/(n-1),s['robin_'+k],wall[4*i:4*i+4]) for i,(k,n) in enumerate(zip('xy',(nx,ny))))
    a = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    rhs = s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
    ij = np.asarray(s['sensor_ij'])
    u = [solve(a+shift*np.eye(nx*ny),rhs,check_finite=False).T.reshape(-1,ny,nx).transpose(0,2,1) for shift in s['mu_shifts']]
    return np.concatenate([v[:,ij[:,0],ij[:,1]].T for v in u]), u[-1]

def observations(s):
    c = s['sensor_calibration_sweeps']; x = c['ref_rosette']; r = c['raw_rosette']
    b = np.linalg.lstsq(np.column_stack((np.ones(len(x)),x)),r,rcond=None)[0]
    return solve(b[1:].T,(s['y_obs_raw'].reshape(-1,4)-b[0]).T).T.ravel()

def fit(s, A, y, rows=None):
    if rows is None: rows = np.arange(len(y))
    Z = s['Z_nuisance'][rows]; a = A[rows]; b = y[rows]
    zp = solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T,assume_a='pos')
    at = a-Z@(zp@a); bt = b-Z@(zp@b)
    H = at.T@at+s['source_ridge']*np.eye(a.shape[1]); c = at.T@bt
    cap, mass = s['u_box'],s['total_budget']
    opt = minimize(lambda w: (0.5*w@H@w-c@w,H@w-c),np.full(len(c),mass/len(c)),jac=True,method='SLSQP',bounds=[(0,cap)]*len(c),constraints=[{'type':'eq','fun':lambda w:w.sum()-mass,'jac':lambda w:np.ones(len(c))}],options={'ftol':1e-14,'maxiter':200})
    w=opt.x
    return w, zp@(b-a@w), at, bt
