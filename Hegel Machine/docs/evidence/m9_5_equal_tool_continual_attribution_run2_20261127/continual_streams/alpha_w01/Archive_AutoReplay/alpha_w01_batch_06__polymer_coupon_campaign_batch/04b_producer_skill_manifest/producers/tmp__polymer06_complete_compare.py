import importlib.util, pickle
import numpy as np
p = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', p + 'alpha_w01_batch_06_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
r, = pickle.load(open(p + 'public_alpha_w01_batch_06.pkl', 'rb')); d = r['dataset']; tr = d['train']; te = d['test']; g = r['gt']
solve_complete = m.solve_alpha_w01_batch_06
source = open(p + 'alpha_w01_batch_06_solver.py').read()
start = source.index('    complete = np.isfinite(train[')
end = source.index('    taus, w = _fit', start)
namespace = {}
exec(compile(source[:start] + source[end:], m.__file__, 'exec'), namespace)
solve_all = namespace['solve_alpha_w01_batch_06']

def take(b, k): return {name: v[k] for name, v in b.items()}
def rm(e): return np.sqrt(np.mean(e * e))
def mad(e): return 1.4826 * np.median(np.abs(e - np.median(e)))
maps = m._load_cells(d['calibration_load_cells']); dt = d['past_dt']; ys = d['yield_stress']
for key, run in [('complete (current)', solve_complete), ('all coupons', solve_all)]:
    print(key)
    o = run(d)
    print('parameters',o['estimated_taus'],o['hereditary_weights'],o['viscoplastic_weights'])
    ec = o['predicted_strain'] - g['te_clean']; er = o['initial_rate_pred'] - g['te_true_rate0']; _, ah = m._alignment(tr,te); h,hold,*_ = m._inputs(te,ah,maps,dt,ys)
    complete = np.isfinite(te['coupon_features_raw']).all(1); above = (np.abs(h).max(1)>ys)|(np.abs(hold)>ys)
    groups = [(te['lab_id']==lab)&(above==a)&(complete==c) for lab in range(4) for a in (False,True) for c in (False,True)]; groups = [k for k in groups if k.any()]
    print('clean pooled curve/rate, worst slice curve/rate, last time:',rm(ec),rm(er),max(rm(ec[k]) for k in groups),max(rm(er[k]) for k in groups),rm(ec[:,-1]))
    for lab in range(4):
        val = tr['lab_id']==lab; va=take(tr,val)
        q=run({**d,'train':take(tr,~val),'test':va})
        good=np.isfinite(va['coupon_features_raw']).all(1)
        e=q['predicted_strain']-va['strain_curves_obs']; v=q['initial_rate_pred']-va['instant_rate_obs']
        h, hold, *_=m._inputs(va,np.zeros((val.sum(),4)),maps,dt,ys); over=(np.abs(h).max(1)>ys)|(np.abs(hold)>ys)
        for flag in (False,True):
            k=good&(over==flag)
            print('held-out complete lab, yield, n, curve MAD, final median absolute error, rate RMSE',lab,int(flag),k.sum(),*(round(x,6) for x in (mad(e[k]),np.median(np.abs(e[k,-1])),rm(v[k]))))
