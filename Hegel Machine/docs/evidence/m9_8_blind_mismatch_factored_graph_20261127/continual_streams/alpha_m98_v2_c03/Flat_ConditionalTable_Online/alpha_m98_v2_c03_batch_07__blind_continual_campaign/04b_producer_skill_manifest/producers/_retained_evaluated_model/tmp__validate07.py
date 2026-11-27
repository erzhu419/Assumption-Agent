import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=runpy.run_path('/holdout_task/batch_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; v=i['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; tr=d['train']; te=d['test']; loads=d['calibration_load_cells']; m=s['_calibration'](loads)
def obs(data): return np.column_stack((data['strain_curves_obs']-data['strain_0'][:,None],data['instant_rate_obs']))
def stack(p): return np.column_stack(p)
def metrics(r): return np.array([np.median(abs(r[:,:-1])),np.median(abs(r[:,-1])),np.quantile(abs(r[:,:-1]),.9),np.quantile(abs(r[:,-1]),.9)])
c,tc=s['_prepare'](d); y=obs(tr); z=obs(v); dc=s['_coupon'](v,s['_features'](v,v)[:len(z)],m)
u,w=s['_fit'](c,dt,t,y)
print('material',u,w)
print('chamber independently normalized panel',metrics(stack(s['_chamber_predict'](dc,dt,t,u,w))-z))
for lab in np.unique(tr['lab_id']):
 keep=tr['lab_id']!=lab; test=v['lab_id']==lab; ub,wb=s['_fit'](tuple(q[keep] for q in c),dt,t,y[keep]); ec=s['_design']((dc[0],dc[1],np.zeros_like(dc[2])),dt,t,ub)@wb; f=dc[2]@wb[2:];
 def pred(tau): return ec+f[:,None]*np.r_[tau*-np.expm1(-t/tau),1.]
 tune=least_squares(lambda q:(pred(np.exp(q[0]))[~test]-z[~test]).ravel(),[0.],loss='cauchy',f_scale=.01)
 print('held lab',lab,'n',np.sum(test),'tau',ub,'bench',metrics((s['_design'](c,dt,t,ub)@wb-y)[~keep]),'chamber unit',metrics((pred(1)-z)[test]),'chamber fitted',metrics((pred(np.exp(tune.x[0]))-z)[test]),'fitted drift tau',np.exp(tune.x[0]))
x=loads['true_channel_loads']; raw=loads['bridge_raw']; keep=np.max(abs(raw-np.median(raw,axis=0)),axis=1)<8
for k in range(4):
 ids=np.flatnonzero(keep); fit=ids[np.arange(len(ids))%4!=k]; test=ids[np.arange(len(ids))%4==k]; model=s['_calibration']({'true_channel_loads':x[fit], 'bridge_raw':raw[fit]}); pred=s['_loads'](raw[test],model); print('bridge fold',k,'channel max',np.max(abs(pred-x[test])),'total max',np.max(abs((pred-x[test]).sum(axis=1))))
print('deployment peak abs stress',max(np.max(abs(tc[0])),np.max(abs(tc[1]))))
