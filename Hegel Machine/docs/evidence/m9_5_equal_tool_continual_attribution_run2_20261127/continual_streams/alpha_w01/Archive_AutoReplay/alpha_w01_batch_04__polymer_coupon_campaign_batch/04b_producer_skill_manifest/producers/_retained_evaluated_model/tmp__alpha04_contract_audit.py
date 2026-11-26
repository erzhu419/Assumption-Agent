import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_w01_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open(root+'public_alpha_w01_batch_04.pkl','rb'))[0]['dataset']
b = d['train']; panel = d['calibration_load_cells']; time = d['eval_times']; dt = d['past_dt']; sy = d['yield_stress']; labs = b['lab_id']
maps = s._load_cells(panel)
x = s._alignment(b,d['test'])[0]
out = s.solve_alpha_w01_batch_04(d)
tau = out['estimated_taus']; w = np.r_[out['hereditary_weights'],out['viscoplastic_weights']]
inputs = s._inputs(b,x,maps,dt,sy)
a,r = s._design(tau,inputs,dt,time)
# Independent step-by-step update of each branch on the observed histories.
gain,bias = maps[labs].T
hist = gain[:,None]*b['past_stress_cmd'] + bias[:,None]
hold = gain*b['stress_hold_cmd'] + bias
state = np.zeros((len(labs),2)); work = np.zeros(len(labs)); qa = abs(hold)>sy
for j,step in enumerate(dt):
    decay = np.exp(-step/tau)
    state = decay*state + (1-decay)*hist[:,j,None]
    work += step*np.where(abs(hist[:,j])>sy,(abs(hist[:,j])-sy)**2,0.)
hold_drive = np.where(qa, np.sign(hold)*(abs(hold)-sy)**2,0.)
for k,t in enumerate(time):
    decay = np.exp(-t/tau)
    expected = np.column_stack(((hold[:,None]-state)*(1-decay), np.ones(len(labs))*t, x*t, hold_drive*t, work*hold*(1-decay[0])))
    np.testing.assert_allclose(a[:,k],expected,atol=1e-13,rtol=1e-12)
np.testing.assert_allclose(r,np.column_stack(((hold[:,None]-state)/tau,np.ones(len(labs)),x,hold_drive,work*hold/tau[0])),atol=1e-13,rtol=1e-12)
print('Independent recurrence and derivative passed; observed activation counts (history, hold):', np.unique(np.c_[work>0,qa],axis=0,return_counts=True))

# Hold onset out when diagnosing nuisance offsets, and use it only to score
# independent future strain when testing the missing-feature update.
for held in ([0],[1],[2],[3],[0,1],[2,3]):
    mask = np.isin(labs,held)
    train = {k:np.asarray(v)[~mask] for k,v in b.items()}
    test = {k:np.asarray(v)[mask] for k,v in b.items() if k not in ('instant_rate_obs','strain_curves_obs')}
    blind = s.solve_alpha_w01_batch_04({**d,'train':train,'test':test})
    updated = s.solve_alpha_w01_batch_04({**d,'train':train,'test':{**test,'instant_rate_obs':b['instant_rate_obs'][mask]}})
    ra = blind['initial_rate_pred'] - b['instant_rate_obs'][mask]
    ea = updated['predicted_strain'] - b['strain_curves_obs'][mask]
    complete = np.isfinite(test['coupon_features_raw']).all(axis=1)
    for lab in held:
        for status in (False,True):
            m = (test['lab_id']==lab) & complete & (((work[mask]>0)|qa[mask])==status)
            if m.any():
                print('held',held,'lab',lab,'plastic',status,'n',m.sum(),'rate bias/MAD and late strain median/MAE',np.round([np.median(ra[m]),np.median(np.abs(ra[m]-np.median(ra[m]))),np.median(ea[m,-1]),np.mean(np.abs(ea[m,-1]))],6))
    m = ~complete
    if m.any():
        old = blind['predicted_strain'][m] - b['strain_curves_obs'][mask][m]
        new = ea[m]
        assert np.abs(new).mean() < np.abs(old).mean()
        print('held',held,'imputed curve and late MAE',np.round([np.abs(new).mean(),np.abs(new[:,-1]).mean()],6))

# On either wing, combine sparse and severely corrupted current checks.
for wing in ([0,1],[2,3]):
    pl = panel['lab_id']; keep = np.ones(len(pl),dtype=bool); y = panel['true_stress'].copy()
    for lab in wing:
        ids=np.flatnonzero(pl==lab); keep[ids[1:]]=False; y[ids[0]]+=.15
    sparse = {k:np.asarray(v)[keep] for k,v in {**panel,'true_stress':y}.items()}
    alternate = s.solve_alpha_w01_batch_04({**d,'calibration_load_cells':sparse})
    max_d = {k:float(np.max(np.abs(alternate[k]-out[k]))) for k in out}
    print('wing',wing,'corrupted spot-check max output changes',max_d)
