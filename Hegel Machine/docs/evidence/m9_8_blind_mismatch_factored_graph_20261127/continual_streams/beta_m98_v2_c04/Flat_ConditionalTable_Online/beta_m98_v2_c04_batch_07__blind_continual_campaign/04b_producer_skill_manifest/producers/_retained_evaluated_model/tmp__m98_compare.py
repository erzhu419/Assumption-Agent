import glob, importlib.util, pickle
import numpy as np
p = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = p['study']; cal = s['sensor_calibration_sweeps']
ref, raw = cal['ref_rosette'], cal['raw_rosette']
Z = s['Z_nuisance']; P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
files = sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py'))
mods = {}
for f in files:
    name = f.rsplit('/',1)[-1].split('_solver')[0].replace('beta_m98_v2_c04_','')
    spec = importlib.util.spec_from_file_location(name,f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[name] = m
bank = cal['single_bank_sweeps']; print('bank sweep contents:', {k:{l:np.shape(v) for l,v in a.items()} if isinstance(a,dict) else np.shape(a) for k,a in bank.items()} if isinstance(bank,dict) else repr(bank)[:300]); print('ref range', ref.min(0), ref.max(0)); print('bank groups', np.unique(s['row_bank_id'].reshape(-1,4),axis=0))
for ck in ('s00','batch_04','batch_06'):
    m = mods[ck]; errors=[]
    for fold in range(5):
        test = np.arange(len(ref))%5==fold
        t = dict(s,sensor_calibration_sweeps=dict(cal,ref_rosette=ref[~test],raw_rosette=raw[~test]),y_obs_raw=raw[test].ravel())
        pred = m._calibrate(t).reshape(-1,4); errors.extend(pred-ref[test])
    e=np.asarray(errors); print('cal',ck,'channel RMSE',np.sqrt(np.mean(e**2,axis=0)),'max',np.max(abs(e)))
    for bk,b in mods.items():
        S,U = b._forward(s); scores=[]
        for pulse in p['campaign_diagnostics']:
            y = m._calibrate(dict(s,y_obs_raw=pulse['y_pulse_raw'])); d=y-S@pulse['w_pulse']; scores.append((np.linalg.norm(d)/np.linalg.norm(y), np.linalg.norm(d-Z@(P@d))/np.linalg.norm(y)))
        print(ck,bk,'pulse raw/projected relative',np.round(scores,6).tolist())
