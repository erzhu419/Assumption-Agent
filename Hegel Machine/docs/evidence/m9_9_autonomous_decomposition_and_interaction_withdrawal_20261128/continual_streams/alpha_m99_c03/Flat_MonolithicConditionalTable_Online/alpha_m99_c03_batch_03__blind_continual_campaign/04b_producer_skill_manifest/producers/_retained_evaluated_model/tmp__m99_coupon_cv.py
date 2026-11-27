import importlib.util, pickle, numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

with open('/holdout_task/public_batch.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
a, b = d['train'], d['test']; t, dt = d['eval_times'], d['past_dt']; n = len(a['lab_id'])
y = a['strain_curves_obs'] - a['strain_0'][:, None]; rate = a['instant_rate_obs']
for label, path in [('affine', '/tmp/hegel_continual_archive/alpha_m99_c03_s00_solver.py'), ('cyclic', '/holdout_task/batch_solver.py')]:
    m = load(path)
    x = m._features(a,b)
    inputs = m._coupon_inputs(a,x[:n],m._calibration(d['calibration_load_cells']))
    p,w = m._fit(inputs,dt,t,y,rate)
    print(label,'full tau',p,'weights',w)
    for lab in np.unique(a['lab_id']):
        valid = a['lab_id']==lab
        p,w=m._fit(tuple(v[~valid] for v in inputs),dt,t,y[~valid],rate[~valid]); c,r=m._design(p,tuple(v[valid] for v in inputs),dt,t)
        ce=np.abs(c@w-y[valid]); re=np.abs(r@w-rate[valid]);
        print(label,'lab',lab,'tau',p.round(5),'curve median/p90',np.nanmedian(ce),np.nanquantile(ce,.9),'rate median/p90',np.nanmedian(re),np.nanquantile(re,.9),'horizon median worst',np.nanmedian(ce,axis=0).max())
