import runpy
p=runpy.run_path('/tmp/calibration_audit.py'); globals().update({k:p[k] for k in ('np','m','i','s','sw','X','Y','S','U')})
B=sw['single_bank_sweeps']; gains=[]; offsets=[]
for j in range(4):
    mask=B['bank_id']==j
    r=np.r_[X[:,j], B['ref_signal'][mask]]; y=np.r_[Y[:,j],B['raw_readout'][mask]]
    c=np.linalg.lstsq(np.c_[np.ones(len(r)),r],y,rcond=None)[0]; offsets.append(c[0]); gains.append(c[1])
def diagonal(study):
    return ((study['y_obs_raw'].reshape(-1,4)-offsets)/gains).ravel()
for label,f in [('affine',m._calibrate),('pooled diagonal',diagonal)]:
    E=np.array([S@q['w_pulse']-f(dict(s,y_obs_raw=q['y_pulse_raw'])) for q in i['campaign_diagnostics']])
    print(label,'worst calibrated pulse/shift RMSE',np.sqrt(np.mean(E.reshape(len(E),len(s['mu_shifts']),-1)**2,axis=2)).max())
Z=s['Z_nuisance']; y=m._calibrate(s); zridge=s['nuisance_ridge']; ridge=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; w=m.solve_batch(s)['weights']
for use_z in (False,True):
    err=[]; drift=[]
    for bank in range(4):
        v=np.arange(len(y))%4==bank; t=~v
        # Fit and profile on three banks, then predict the held-out bank in calibrated units.
        A=Z[t]
        P=np.linalg.solve(A.T@A+zridge*np.eye(Z.shape[1]),A.T)
        D=S[t]-A@(P@S[t]) if use_z else S[t]; b=y[t]-A@(P@y[t]) if use_z else y[t]
        wt,_=m._box_budget(D.T@D+ridge*np.eye(S.shape[1]),-D.T@b,cap,budget)
        gamma=P@(y[t]-S[t]@wt) if use_z else np.zeros(Z.shape[1]); e=S[v]@wt+Z[v]@gamma-y[v]
        err.append(np.sqrt(np.mean(e**2))); drift.append(np.linalg.norm(wt-w)/np.linalg.norm(w))
    print('profiled nuisance',use_z,'held-bank RMSE',err,'relative weight drift',drift)
