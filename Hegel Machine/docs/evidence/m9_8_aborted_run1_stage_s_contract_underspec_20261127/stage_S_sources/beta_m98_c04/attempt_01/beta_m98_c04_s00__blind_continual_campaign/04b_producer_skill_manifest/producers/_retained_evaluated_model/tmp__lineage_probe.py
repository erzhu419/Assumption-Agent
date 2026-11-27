import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from transport_refine import *
from scipy.optimize import minimize
for a,B in zip('xy',bases):
 k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(len(k)-1); print(a,'robin',s['robin_'+a],'h',h,'k',np.round(k,6),'v',np.round(v,6),'base wall diagonals',B.diagonal()[[0,-1]],'base wall inward',B[0,1],B[-1,-2]); print('wall coefficients normalized',np.array([.491511,.491511] if a=='x' else [-.07352,.07352])*h*h/k[[0,-1]])
c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Yc=c['raw_rosette']; print('sweep shape',X.shape,'max identity error',abs(Yc-X).max()); print('budget/box/ridges',s['total_budget'],s['u_box'],s['source_ridge'],s['nuisance_ridge']); print('telemetry first four rows',np.round(s['aux_manifold_telemetry'][:4],4)); print('Z stats',np.linalg.norm(Z,axis=0))
# Known-pulse wall law, used only to estimate the observation noise floor.
r=np.array([2.916668,3.698159,6.627348,5.234422]); ds=[.491511,.491511,-.07352,.07352]; axes=[]
for t,B in enumerate(bases):
 a=B.toarray();a[0,1]+=ds[2*t];a[-1,-2]+=ds[2*t+1];axes.append(sp.csc_matrix(a))
from transport_probe import predict
F=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
S=pred(r,axes); z=np.c_[S,Z]; yy=s['y_obs_raw']; M=S.shape[1]; w=np.linalg.lstsq(z,yy,rcond=None)[0]; print('unconstrained joint fit',np.round(w,8),'sum',sum(w[:M]),'RMS',np.std(z@w-yy)); fit=minimize(lambda w:0.5*np.sum((z@w-yy)**2),w,jac=lambda w:z.T@(z@w-yy),bounds=list(zip(np.zeros(M),np.broadcast_to(s['u_box'],M)))+[(None,None)]*Z.shape[1],constraints={'type':'eq','fun':lambda w:sum(w[:M])-s['total_budget'],'jac':lambda w:np.r_[np.ones(M),np.zeros(Z.shape[1])]},method='SLSQP',options={'ftol':1e-14}); print('box joint fit',np.round(fit.x,8),'RMS',np.std(z@fit.x-yy))
