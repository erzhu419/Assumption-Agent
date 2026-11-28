import pickle, runpy
import numpy as np
p = '/holdout_task/'
d = pickle.load(open(p + 'public_batch.pkl', 'rb'))[0]
s = d['study']; m = runpy.run_path(p + 'batch_solver.py')
S,_ = m['_responses'](s,m['_operator'](s))
c = s['sensor_calibration_sweeps']; X = c['ref_rosette']; Y = c['raw_rosette']; b = c['single_bank_sweeps']
R = np.c_[np.ones(len(X)),X]
T = np.zeros((len(b['bank_id']),5)); T[:,0] = 1
T[np.arange(len(T)),1+np.asarray(b['bank_id'])] = b['ref_signal']
Z = s['Z_nuisance']; P = np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
A = P@S; H = A.T@A+s['source_ridge']*np.eye(S.shape[1]); Q = np.linalg.qr(np.ones((S.shape[1],1)),mode='complete')[0][:,1:]
for combine in (False, True):
    metrics = []
    for fold in range(6):
        test = np.array_split(np.arange(len(R)),5)[fold] if fold<5 else np.arange(len(R))
        train = np.setdiff1d(np.arange(len(R)),test) if fold<5 else np.arange(len(R))
        fits=[]; bank_err=[]
        for k in range(4):
            ids=np.flatnonzero(b['bank_id']==k); kt=np.arange(len(ids))%5!=fold; a=R[train]; y=Y[train,k]
            if combine:
                a=np.vstack([a,T[ids[kt]]]); y=np.r_[y,b['raw_readout'][ids[kt]]]
            fit=np.linalg.lstsq(a,y,rcond=None)[0]; fits.append(fit)
            if fold<5: bank_err.append(np.sqrt(np.mean((T[ids[~kt]]@fit-b['raw_readout'][ids[~kt]])**2)))
        B=np.column_stack(fits); row=[]
        for pulse in d['campaign_diagnostics']:
            observed=np.linalg.solve(B[1:].T,(pulse['y_pulse_raw'].reshape(-1,4)-B[0]).T).T.ravel(); e=observed-S@pulse['w_pulse']; shift=e.reshape(len(s['mu_shifts']),-1,4)
            dw=Q@np.linalg.solve(Q.T@H@Q,Q.T@A.T@P@e)
            row.append([np.max(np.sqrt(np.mean(shift**2,axis=(1,2)))),np.max(np.sqrt(np.mean(shift**2,axis=1))),np.linalg.norm(dw)])
        metrics.append([np.max(np.sqrt(np.mean((R[test]@B-Y[test])**2,axis=0))),max(bank_err,default=0),*np.max(row,axis=0)])
    print(('joint sweeps' if combine else 'rosette only'), 'metrics: rosette RMSE, bank RMSE, pulse shift RMSE, pulse shift/channel RMSE, projected weight sensitivity')
    print('  worst outer fold:',np.max(metrics[:5],axis=0)); print('  full fit, independent pulses:',metrics[-1][2:])
