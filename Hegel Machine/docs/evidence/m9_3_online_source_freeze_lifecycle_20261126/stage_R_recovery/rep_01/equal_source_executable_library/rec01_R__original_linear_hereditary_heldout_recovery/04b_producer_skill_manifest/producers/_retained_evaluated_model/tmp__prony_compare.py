import importlib.util, pickle, numpy as np
from scipy.stats import median_abs_deviation

def load(path):
    spec = importlib.util.spec_from_file_location('candidate', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.solve_hereditary_prony_recovery

root='/holdout_task/'
r=pickle.load(open(root+'public_rec01_R_datasets.pkl','rb'))[0]; d=r['dataset']; g=r['gt']; s=d['test']; tr=d['train']
rmse=lambda x: np.mean(x*x)**.5
for path in [root+'hereditary_prony_recovery_solver.py','/tmp/prony_challenger.py']:
 f=load(path); o=f(d); e=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']; valid=np.isfinite(s['coupon_features_raw']).all(axis=1)
 print(path, 'taus',o['estimated_taus'], 'weights',o['hereditary_weights'].round(6))
 cells=[(s['lab_id']==j)&(s['stratum']==k) for j in range(4) for k in range(3)]
 print('overall/worst cell/last horizon curve',rmse(e),max(rmse(e[i]) for i in cells),rmse(e[:,-1]),'overall/worst cell rate',rmse(er),max(rmse(er[i]) for i in cells))
 print('complete/missing curve and rate',[(rmse(e[i]),rmse(er[i])) for i in (valid,~valid)])
 for lab in range(4):
    i=tr['lab_id']==lab; d2={**d,'train':{k:v[~i] for k,v in tr.items()},'test':{k:v for k,v in s.items()}}; oo=f(d2); a=s['lab_id']==lab
    print('LOLO',lab,'tau max rel err',max(abs(oo['estimated_taus']/g['taus']-1)), 'curve',rmse(oo['predicted_strain'][a]-g['te_clean'][a]),'rate',rmse(oo['initial_rate_pred'][a]-g['te_true_rate0'][a]))
