exec(open('/tmp/m98_probe.py').read().split("print('baseline chamber'")[0])
Z=old._design((np.zeros_like(past),hold,drift),dt,t,tau)@b
R=Y-Z
np.set_printoptions(precision=3,suppress=True,linewidth=150)
print('time',t); print('past, hold, standardized features, reset residual per coupon:')
for i in range(len(hold)): print(i,'lab/str',ch['lab_id'][i],ch['stratum'][i],past[i],hold[i].round(3),x[i],R[i])
K=np.column_stack([-np.expm1(-t[:,None]/tau),t]); print('SVD reset residual',np.linalg.svd(R[:,:-1],compute_uv=False)); print('SVD right:',np.linalg.svd(R[:,:-1],full_matrices=False)[2][:3]); print('per coupon curve residual fit to intrinsic kernels and drift, actual rate residual and implied rate:')
c=np.linalg.lstsq(K,R[:,:-1].T,rcond=None)[0].T
print(np.c_[c,R[:,-1],c@np.r_[1/tau,1],np.sqrt(np.mean((c@K.T-R[:,:-1])**2,axis=1))]); print('reset rates residual vs candidate features'); q=np.column_stack((hold,past,drift,hold-past[:,-1],past[:,-1]-past[:,-2])); print(np.corrcoef(np.c_[R[:,-1],R[:,0],R[:,-2],q].T)[:3,3:]); print('cal sweeps',d['calibration_load_cells']['single_channel_sweeps'])
