import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('s', '/holdout_task/alpha_m96_w03_batch_04_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_alpha_m96_w03_batch_04.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']; t = d['eval_times']; args = d['past_dt'], t; ys = d['yield_stress']; xt, _ = s._align(tr,te)
missing = ~np.isfinite(tr['coupon_features_raw']); lab = tr['lab_id']; strata = tr['stratum']; folds = np.zeros(len(lab),int)
for j in range(4):
 for flag in (False, True):
  ids = np.flatnonzero((lab == j) & (missing.any(axis=1) == flag)); np.random.default_rng(173+j).shuffle(ids); folds[ids] = np.arange(len(ids)) % 5
E = np.zeros((3,len(lab),len(t))); R = np.zeros((3,len(lab))); smoke = []
def predict(block, x, cal, p):
 h = s._Hold(*args, block, x, cal, ys); return s._predict(h,p)
def pair(block):
 return np.column_stack((block['strain_curves_obs']-block['strain_0'][:,None],block['instant_rate_obs']))
def mad(x):
 return 1.4826*np.median(abs(x-np.median(x)))
for f in range(5):
 m = folds == f
 train = {k:v[~m] for k,v in tr.items()}; val = {k:v[m] for k,v in tr.items()}
 fit = dict(d, train=train); cal,p = s._estimate(fit,xt[~m]); print('fold parameters',f,np.round(p[:2],4),np.round(p[-2:],4))
 residual = pair(train)-predict(train,xt[~m],cal,p); complete = ~missing[~m].any(axis=1)
 noise = mad(residual[complete,-1]-np.median(residual[complete,:-1]/t,axis=1)); print('rate noise',round(noise,6))
 variance = np.array([np.var(xt[~m & ~missing[:,j],j]) for j in range(4)])
 v = missing[m] @ (variance * p[5:9]**2); w = v/(v+noise**2)
 diff = pair(val)-predict(val,xt[m],cal,p)
 corrections = np.stack([np.zeros(m.sum()), w*diff[:,-1], missing[m].any(axis=1)*diff[:,-1]])
 E[:,m] = diff[None,:,:-1]-corrections[:,:,None]*t
 R[:,m] = diff[None,:,-1]-corrections
for flag in (False,True):
 for j in [-1,0,1,2,3]:
  m = (missing.any(axis=1)==flag)&((lab==j)|(j==-1))
  if not m.any(): continue
  # A median across time rejects individual curve glitches without deleting difficult coupons.
  er = np.median(E[:,m]/t, axis=2)
  print('missing, lab, n',flag,j,m.sum(),'curve drift MAE',np.round(np.mean(abs(er),axis=1),5),'drift abs q90',np.round(np.quantile(abs(er),.9,axis=1),5),'rate abs median against noisy held-out data',np.round(np.median(abs(R[:,m]),axis=1),5))
