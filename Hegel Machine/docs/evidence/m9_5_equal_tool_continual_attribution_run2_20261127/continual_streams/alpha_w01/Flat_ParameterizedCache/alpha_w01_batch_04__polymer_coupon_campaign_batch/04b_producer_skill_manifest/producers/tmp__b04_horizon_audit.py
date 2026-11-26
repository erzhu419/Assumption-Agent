import importlib.util, pathlib, pickle, types
import numpy as np
p=pathlib.Path('/holdout_task/alpha_w01_batch_04_solver.py')
spec=importlib.util.spec_from_file_location('baseline',p)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
source=p.read_text()
old='    for m in (is_curve, ~is_curve):\n        if np.any(m):\n            scale[m] = _mad(e[m])'
new='    channel = np.r_[np.tile(np.arange(len(time)), len(rates)), np.full(len(rates),len(time))][valid]\n    for j in range(len(time)+1):\n        m = channel == j\n        if np.any(m):\n            scale[m] = _mad(e[m])'
assert source.count(old)==1
q=types.ModuleType('challenger'); exec(compile(source.replace(old,new),'/tmp/b04_horizon_challenger.py','exec'),q.__dict__)
z=pickle.load(open(p.parent/'public_alpha_w01_batch_04.pkl','rb'))[0]; d=z['dataset']; tr,te=d['train'],d['test']; dt,t,sy=(d[k] for k in ('past_dt','eval_times','yield_stress'))
x,_,complete,_=s._align(tr,te); maps=s._calibrate(d['calibration_load_cells']); h,hold=s._stress(tr,maps); ov,w=s._plastic(h,hold,dt,sy)
for mod in (s,q):
    err=np.empty_like(tr['strain_curves_obs']); re=np.empty(len(err))
    for lab in range(4):
        b=tr['lab_id']==lab; a=~b
        tau,theta=mod._fit({k:v[a] for k,v in tr.items()},x[a],complete[a],maps,dt,t,sy)
        A,B=s._design(tau,h[b],hold[b],x[b],dt,t,ov[b],w[b]); rate=B@theta
        re[b]=rate-tr['instant_rate_obs'][b]
        err[b]=tr['strain_0'][b,None]+A@theta-np.where(complete[b],0,re[b])[:,None]*t-tr['strain_curves_obs'][b]
    labels=np.column_stack((tr['lab_id'],tr['stratum']))
    cells=[]
    for cell in np.unique(labels,axis=0):
        b=np.all(labels==cell,axis=1)
        cells.append((tuple(cell),round(float(np.median(abs(err[b]))),6),round(float(np.median(err[b,-1])),6)))
    print(mod.__name__,'lab-stratum (curve MedAE, final signed median)',cells)
    print(mod.__name__,'worst lab curve MedAE',max(np.median(abs(err[tr['lab_id']==i])) for i in range(4)),'worst lab complete-rate MedAE',max(np.median(abs(re[(tr['lab_id']==i)&complete])) for i in range(4)))
    o=mod.solve_alpha_w01_batch_04(d); g=z['gt']; e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']
    print(mod.__name__,'public worst lab curve/rate RMSE',*[max(np.sqrt(np.mean(v[te['lab_id']==i]**2)) for i in range(4)) for v in (e,r)],'taus',o['estimated_taus'],'plastic',o['viscoplastic_weights'])
