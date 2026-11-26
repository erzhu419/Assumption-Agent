import importlib.util
import pickle
import json
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'alpha_w02_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
case = pickle.load(open(root + 'public_alpha_w02_batch_04.pkl','rb'))[0]
d = case['dataset']; tr = d['train']; dt = d['past_dt']; time = d['eval_times']
cal = s._calibrate(d['calibration_load_cells'])
print('current calibration:', np.round(cal, 6).tolist())
for a in s._archive_checks(s.Path('/tmp/hegel_continual_archive')):
    rows = []
    for lab in range(4):
        fits = s._consensus(*s._checks(a, lab))
        rows.append([(np.round(f[0], 5).tolist(), int(f[1]), round(f[2], 2)) for f in fits][:3])
    print('archive calibration', a['episode_short_name'], rows)

y = np.column_stack((tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']))
err = {0: np.empty_like(y), 1: np.empty_like(y)}
stress = s._stress(tr, cal)
ov, hist = s._plastic(stress,dt,d['yield_stress'])
for lab in range(4):
    hold = tr['lab_id']==lab
    train = {k:v[~hold] for k,v in tr.items()}
    test = {k:np.concatenate((v[hold],d['test'][k])) for k,v in tr.items() if k in d['test'] and k not in ('instant_rate_obs','strain_curves_obs')}
    n = (~hold).sum(); nh = hold.sum()
    feat = s._features(train,test)
    st_train = tuple(a[~hold] for a in stress); st_hold = tuple(a[hold] for a in stress)
    for full in err:
        ptrain = (ov[~hold],hist[~hold]) if full else None
        phold = (ov[hold],hist[hold]) if full else None
        tau, coef = s._fit(st_train,feat[:n],ptrain,dt,time,y[~hold])
        err[full][hold] = s._design(st_hold,feat[n:n+nh],phold,dt,time,tau)@coef-y[hold]
        print('fold',lab,'plastic',full,'tau',np.round(tau,5),'weights',np.round(coef,6))

def metrics(e,m):
    z=e[m]; return np.round([np.sqrt(np.mean(z[:,:-1]**2)),np.sqrt(np.mean(z[:,-1]**2)),np.max(np.sqrt(np.mean(z[:,:-1]**2,axis=0)))],6).tolist()
for label,m in [(f'lab {i}',tr['lab_id']==i) for i in range(4)]+[('positive hold over',ov>0),('negative hold over',ov<0),('history only',(ov==0)&(hist!=0)),('no plastic',(ov==0)&(hist==0))]:
    if m.any(): print(label,int(m.sum()),'linear',metrics(err[0],m),'unified',metrics(err[1],m))
res=s.solve_alpha_w02_batch_04(d)
truth=case['gt']; print('public reference keys:',list(truth))
for k,p in res.items():
    if k in truth and np.shape(truth[k])==np.shape(p):
        diff=np.asarray(p)-np.asarray(truth[k]); print('public reference',k,'rmse',np.sqrt(np.mean(diff**2)),'maxabs',np.max(np.abs(diff)))
    if np.size(p)<10: print('final',k,np.round(p,7))
