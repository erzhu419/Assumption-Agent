import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']
cal = d['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
a = m._bridge_terms(x)
ids = np.arange(len(x))
# Compare additional pair-difference, rational saturation, and 3-channel terms
# against the same held-out calibration rows, including single-channel anchors.
for mode in ('compact','pair_square','rational','triple'):
    designs = []
    for j in range(4):
        u = x[:,j]; v = x[:,(j+1)%4]; w = x[:,(j+2)%4]
        terms = [a, u**3, u*v]
        if mode == 'pair_square': terms += [(u-v)**2]
        if mode == 'rational': terms += [u**3/(1+(u-v)**2), u*v/(1+(u-v)**2)]
        if mode == 'triple': terms += [u*v*w]
        designs.append(np.column_stack(terms))
    fold = []
    errors = np.empty(len(x))
    for k in range(5):
        test = ids % 5 == k
        keep = ~test
        for _ in range(6):
            pred = np.column_stack([A @ np.linalg.lstsq(A[keep], y[keep,j], rcond=None)[0] for j,A in enumerate(designs)])
            error = np.linalg.norm(pred-y,axis=1)
            keep = (~test) & (error < max(5*np.median(error[keep]),1e-8))
        fold.append(np.median(error[test])); errors[test] = error[test]
    anchor = np.count_nonzero(x,axis=1)==1
    print('bridge',mode,'worst fold median',max(fold),'anchor median',np.median(errors[anchor]),'mixed median',np.median(errors[~anchor]))

tr,te,g = d['train'], d['test'], item['campaign_diagnostics']
dt,t = d['past_dt'],d['eval_times']
c = m._fit_bridge(cal)
X = m._aligned_features(tr,te)
train = m._coupon(tr,X[:len(tr['lab_id'])],c)
group = m._coupon(g,m._aligned_features(g,g)[:len(g['lab_id'])],c)
yc = tr['strain_curves_obs']-tr['strain_0'][:,None]
yr = tr['instant_rate_obs']
gc = g['strain_curves_obs']-g['strain_0'][:,None]
gr = g['instant_rate_obs']
rmse = lambda x: np.sqrt(np.mean(x*x))
for adaptive in (False, True):
    tau,beta = m._fit(dt,t,train,yc,yr,adaptive=adaptive)
    p,r = m._chamber(tau,beta,dt,t,group)
    e,s = p-gc,r-gr
    print('adaptive',adaptive,'tau',tau,'chamber worst lab curve/rate',max(rmse(e[g['lab_id']==lab]) for lab in np.unique(g['lab_id'])),max(rmse(s[g['lab_id']==lab]) for lab in np.unique(g['lab_id'])),'chamber early/late RMSE',rmse(e[:,:3]),rmse(e[:,-3:]))
    held = []
    for lab in np.unique(tr['lab_id']):
        h = tr['lab_id']==lab
        u,v = m._fit(dt,t,tuple(z[~h] for z in train),yc[~h],yr[~h],adaptive=adaptive)
        A,B = m._design(u,dt,t,tuple(z[h] for z in train))
        held.append((np.median(abs(A@v-yc[h])),np.median(abs(B@v-yr[h])),*abs(u/tau-1)))
    print('benchtop worst held-lab curve/rate median and tau relative shift',np.max(held,axis=0))
