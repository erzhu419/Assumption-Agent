import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']; D = m._rosette_features(x)

def keep(tr):
    e = np.abs(D[tr] @ np.linalg.lstsq(D[tr], y[tr], rcond=None)[0] - y[tr]).max(1)
    return tr[e <= max(5*np.median(e),1e-10)]

def fit(tr, te, family):
    out = np.empty((len(te),4))
    for j in range(4):
        p, q = (j-1)%4, (j+1)%4
        X = D if family == 'baseline' else D[:, [0,1+j,1+p,1+q,5+j,9+j]]
        if family == 'rational':
            # Fit an adjacent-channel difference in the denominator.
            z = (x[:,j]-x[:,q])**2
            v = np.linalg.lstsq(np.column_stack((X[tr], -y[tr,j]*z[tr])), y[tr,j], rcond=None)[0]
            den = 1+v[-1]*z[te]
            out[:,j] = X[te] @ v[:-1] / den
        else:
            out[:,j] = X[te] @ np.linalg.lstsq(X[tr], y[tr,j], rcond=None)[0]
    return out

for f in ('baseline','sparse','rational'):
    err = np.empty_like(y)
    for i in range(len(x)):
        tr = keep(np.delete(np.arange(len(x)),i)); err[i] = fit(tr,[i],f)[0]-y[i]
    # Report gross errors separately; no test labels are used in training.
    clean = np.max(np.abs(err),axis=1)<1
    print(f,'gross rows',np.flatnonzero(~clean),'inlier max',np.max(abs(err[clean]),axis=0),'inlier RMSE',np.mean(err[clean]**2,axis=0)**.5)
    if f == 'baseline': assert np.max(abs(err[clean])) < 1e-4

S,U = m._forward(s); ycal = m._calibrate(s); Z = s['Z_nuisance']; ns = len(s['mu_shifts']); n = len(ycal)
fold = np.tile(np.repeat(np.arange(n//ns//4),4),ns)
ridge = s['source_ridge']; rr = s['nuisance_ridge']; cap = s['u_box']; budget = s['total_budget']; e = np.empty(n)
for i in np.unique(fold):
    tr = fold!=i; zz = Z[tr]; P = np.linalg.solve(zz.T@zz+rr*np.eye(Z.shape[1]),zz.T)
    A = S[tr]-zz@(P@S[tr]); b = ycal[tr]-zz@(P@ycal[tr]); w,_ = m._box_budget_qp(A.T@A+ridge*np.eye(S.shape[1]),A.T@b,cap,budget)
    e[~tr] = ycal[~tr]-S[~tr]@w-Z[~tr]@(P@(ycal[tr]-S[tr]@w))
E=e.reshape(ns,-1,4); print('held-out RMSE by shift and bank\n',np.mean(E**2,axis=1)**.5); print('held-out signed means by shift and bank\n',E.mean(axis=1))
assert np.max(np.mean(E**2,axis=(1,2))**.5)<0.012
out=m.solve_batch(s); w=out['weights']; cert=out['kkt_certificate']; nx,ny = s['nx'],s['ny']
# Assemble the specified Robin equations independently, having first combined sources.
def axis(k,v,h,r):
    a=np.zeros((len(k),len(k)))
    for i in range(1,len(k)-1):
        a[i,i-1]=-(k[i-1]+k[i])/(2*h*h)-max(v[i],0)/h
        a[i,i+1]=-(k[i+1]+k[i])/(2*h*h)-max(-v[i],0)/h
        a[i,i]=-a[i,i-1]-a[i,i+1]
    a[0,1]=-k[0]/h**2; a[-1,-2]=-k[-1]/h**2
    a[0,0]=-a[0,1]+r/h; a[-1,-1]=-a[-1,-2]+r/h
    return a
ax,ay=(axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1),s['robin_'+d]) for d in 'xy')
L=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F')+s['mu_shifts'][-1])
r = L@out['terminal_field'].ravel(order='F')-np.tensordot(w,s['source_modes'],axes=(0,0)).ravel(order='F')
P=np.linalg.solve(Z.T@Z+rr*np.eye(Z.shape[1]),Z.T); A=S-Z@(P@S); b=ycal-Z@(P@ycal); g=A.T@(A@w-b)+ridge*w
l,u=cert['lam_lower'],cert['lam_upper']; nu=cert['nu_sum']; print('terminal PDE max residual',abs(r).max(),'stationarity max',abs(g-nu-l+u).max(),'complementarity max',max(abs(l*w).max(),abs(u*(cap-w)).max()))
assert abs(r).max()<1e-10 and abs(g-nu-l+u).max()<1e-10 and max(abs(l*w).max(),abs(u*(cap-w)).max())<1e-10
