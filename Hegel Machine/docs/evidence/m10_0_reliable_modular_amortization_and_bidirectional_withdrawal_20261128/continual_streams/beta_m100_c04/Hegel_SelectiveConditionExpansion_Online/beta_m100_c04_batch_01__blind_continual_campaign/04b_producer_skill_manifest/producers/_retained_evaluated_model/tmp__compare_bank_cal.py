import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
s = item['study']
C = s['sensor_calibration_sweeps']
X = np.c_[np.ones(len(C['ref_rosette'])), C['ref_rosette']]
R = C['raw_rosette']
bank_data = C['single_bank_sweeps']
print('bank keys', list(bank_data))
B, V, T = map(np.asarray, bank_data.values())
B = B.astype(int)
slices = [np.flatnonzero(B == j) for j in range(4)]
D = np.zeros((len(B),5))
D[:,0] = 1
D[np.arange(len(B)),1+B] = V
S, _ = m._forward(s)
P = item['campaign_diagnostics']
W = np.column_stack([p['w_pulse'] for p in P])
Y = np.stack([p['y_pulse_raw'] for p in P]).reshape(len(P),-1,4)
F = (S @ W).T.reshape(Y.shape)

def fit(keep, bank_keep, joint):
    cols = []
    for j in range(4):
        idx = bank_keep & (B == j)
        a, y = X[keep], R[keep,j]
        if joint:
            a, y = np.r_[a,D[idx]], np.r_[y,T[idx]]
        cols.append(np.linalg.lstsq(a,y,rcond=None)[0])
    return np.array(cols).T

def rms(x):
    return np.sqrt(np.mean(x*x))

for joint in (False, True):
    sweep, bank, pulse, worst = [], [], [], []
    for fold, ix in enumerate(np.array_split(np.arange(len(X)), 6)):
        bi = np.concatenate([np.array_split(j,6)[fold] for j in slices])
        keep = ~np.isin(np.arange(len(X)),ix)
        bk = ~np.isin(np.arange(len(B)),bi)
        a = fit(keep,bk,joint)
        sweep.append(rms(X[ix]@a-R[ix]))
        bank.append(rms((D[bi]@a)[np.arange(len(bi)),B[bi]]-T[bi]))
        yc = np.linalg.solve(a[1:].T,(Y-a[0]).reshape(-1,4).T).T.reshape(Y.shape)
        e = (yc-F).reshape(len(P),len(s['mu_shifts']),-1,4)
        pulse.extend(np.sqrt(np.mean(e**2,axis=(1,2,3))))
        worst.append(np.sqrt(np.mean(e**2,axis=2)).max())
    a = fit(np.ones(len(X),bool),np.ones(len(B),bool),joint)
    print('joint',joint,'worst sweep',max(sweep),'worst bank',max(bank),'pooled pulse',rms(np.array(pulse)),'worst pulse slice',max(worst),'offsets/diagonal',a[0],a[1:].diagonal())
