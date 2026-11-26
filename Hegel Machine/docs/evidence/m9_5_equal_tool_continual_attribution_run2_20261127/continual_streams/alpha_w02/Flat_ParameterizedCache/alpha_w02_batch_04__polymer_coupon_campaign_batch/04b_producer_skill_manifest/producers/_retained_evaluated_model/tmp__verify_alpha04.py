import importlib.util, pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_w02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open(root+'public_alpha_w02_batch_04.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; c=d['calibration_load_cells']; tr,te=d['train'],d['test']; dt,t,sy=d['past_dt'],d['eval_times'],d['yield_stress']; cal=s._calibrate(c,sy)
above=(abs(g['te_hold_true'])>sy)|np.any(abs(g['te_hist_true'])>sy,axis=1)
masks={**{f'lab{i}':te['lab_id']==i for i in range(4)},'above':above,'below':~above}
assert above.any() and (~above).any()
def metrics(o):
    e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']
    return np.array([(np.sqrt(np.mean(e[m]**2)),np.sqrt(np.mean(r[m]**2))) for m in masks.values()])
base=s.solve_alpha_w02_batch_04(d); bm=metrics(base); print('public strain/rate RMSE',dict(zip(masks,bm.round(6).tolist()))); assert np.all(np.diff(base['estimated_taus'])>0)
for wing in ((0,1),(2,3)):
    worst=np.zeros(2)
    for scenario in ('none','one','three_emi'):
        keep=~np.isin(c['lab_id'],wing); cc={k:v.copy() for k,v in c.items()}
        for lab in wing:
            ix=np.flatnonzero(c['lab_id']==lab); ix=ix[np.argsort(c['cmd_stress'][ix])]
            if scenario=='one': keep[ix[len(ix)//2]]=True
            if scenario=='three_emi':
                use=ix[3:6]; keep[use]=True; cc['true_stress'][use[-1]]-=.22
        o=s.solve_alpha_w02_batch_04(dict(d,calibration_load_cells={k:v[keep] for k,v in cc.items()})); worst=np.maximum(worst,metrics(o).max(axis=0))
    print('degraded wing',wing,'worst strain/rate RMSE',worst.round(6)); assert np.all(worst < bm.max(axis=0)+.002)
x=s._inputs(tr,s._features(tr,te)[:len(tr['lab_id'])],cal,dt,sy); below=(x[2][:,-1]==0)&(x[3]==0); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; _,w=s._fit(tuple(v[below] for v in x),dt,t,y[below]); assert np.array_equal(w[7:],np.zeros(2)); print('subyield plastic weights',w[7:])
xt=s._inputs(te,s._features(tr,te)[len(tr['lab_id']):],cal,dt,sy); eps=1e-7; a=s._design(xt,dt,np.array([0.,eps]),base['estimated_taus']); w=np.r_[base['hereditary_weights'],base['viscoplastic_weights']]; err=np.max(abs(((a[:,1]-a[:,0])/eps-a[:,2])@w)); assert err<5e-6; print('onset derivative max abs',err)
for lab in range(4):
    ix=np.flatnonzero(c['lab_id']==lab); ix=ix[np.argsort(c['cmd_stress'][ix])]; cmd=np.r_[tr['past_stress_cmd'][tr['lab_id']==lab].ravel(),tr['stress_hold_cmd'][tr['lab_id']==lab]]; worst=0.
    for use in (ix[2:6],ix[[0,3,6,-1]]):
        keep=(c['lab_id']!=lab)|np.isin(np.arange(len(c['lab_id'])),use)
        for bad in use:
            for emi in (-.22,.22):
                cc={k:v.copy() for k,v in c.items()}; cc['true_stress'][bad]+=emi; delta=s._calibrate({k:v[keep] for k,v in cc.items()},sy)[lab]-cal[lab]; worst=max(worst,np.max(abs(cmd*delta[0]+delta[1])))
    print('four-check EMI lab',lab,'max physical stress delta',round(worst,6))
