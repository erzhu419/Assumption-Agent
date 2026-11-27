import importlib.util, pickle, time
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_m96_w02_batch_04_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
case=pickle.load(open(root+'public_alpha_m96_w02_batch_04.pkl','rb'))[0]
d,g=case['dataset'],case['gt']; tr=d['train']; lab=tr['lab_id']; ck=d['calibration_load_cells']; cal=s._calibrate(ck)
def rms(a): return np.sqrt(np.mean(np.square(a)))
def subset(a,m): return {k:v[m] for k,v in a.items()}
def errors(label, dp, y, r):
    t=time.monotonic(); o=s.solve_alpha_m96_w02_batch_04(dp)
    e=o['predicted_strain']-y; f=o['initial_rate_pred']-r
    z=s._Design(dp['test'],np.zeros((len(r),4)),cal,dp['past_dt'],dp['eval_times'],dp['yield_stress']); hist=np.any(abs(z.past)>dp['yield_stress'],axis=1); hold=abs(z.hold)>dp['yield_stress']
    print(label,'time',round(time.monotonic()-t,2),'strain/rate RMS',rms(e),rms(f),'strain/rate medAE',np.median(abs(e)),np.median(abs(f)),'worst horizon RMS',max(rms(e[:,k]) for k in range(e.shape[1])))
    for h,j in [(False,False),(True,False),(False,True),(True,True)]:
        m=(hist==h)&(hold==j)
        if any(m): print('  history/hold',h,j,'n',sum(m),'RMS',rms(e[m]),rms(f[m]))
    return o
out=errors('public clean',d,g['te_clean'],g['te_true_rate0'])
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]: print(k,'fit',out[k],'delta',out[k]-g[j])
for c in range(4):
    m=lab==c
    errors('LOLO '+str(c),dict(d,train=subset(tr,~m),test=subset(tr,m)),tr['strain_curves_obs'][m],tr['instant_rate_obs'][m])
for c in range(4):
    i=np.flatnonzero(ck['lab_id']==c)
    bad={k:v.copy() for k,v in ck.items()}; bad['true_stress'][i[[0,-1]]]+=[.25,-.18]
    other=ck['lab_id']!=c; x=ck['cmd_stress'][i]; far=i[[x.argmin(),x.argmax()]]
    pair=subset(ck,other|np.isin(np.arange(len(other)),far)); one=subset(ck,other| (np.arange(len(other))==far[0]))
    print('calibration errors, lab',c,'max abs parameter delta under two spikes / clean pair / singleton',*[np.max(abs(s._calibrate(q)[c]-cal[c])) for q in (bad,pair,one)])
for k in ('gate','metric','constraint'):
    print(k,[line.strip() for line in open(root+'public_asset_index.json') if k in line.lower()][:12])
