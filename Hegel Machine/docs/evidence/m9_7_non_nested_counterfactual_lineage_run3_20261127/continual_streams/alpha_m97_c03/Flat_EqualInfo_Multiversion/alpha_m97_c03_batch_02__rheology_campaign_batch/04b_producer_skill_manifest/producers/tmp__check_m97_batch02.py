import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(p+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; labs=tr['lab_id']; n=len(labs)
past,hold,x,offset,dt,t=m._prepare(d)
y=np.column_stack((tr['strain_curves_obs']-offset[:n,None],tr['instant_rate_obs']))
for mode in ('soft_l1','cauchy'):
    errs=[]
    for lab in np.unique(labs):
        i=np.flatnonzero(labs!=lab); j=np.flatnonzero(labs==lab)
        tau,w=m._fit((past[i],hold[i],x[i],dt,t),y[i],mode=mode)
        a,da=m._basis(np.log(tau),past[j],hold[j],x[j],dt,t)
        e=abs(a@w-y[j]); errs.append(e)
        print(mode,'lab',lab,'taus',np.round(tau,5),'curve_med',np.median(e[:,:-1]),'worst_time_med',np.max(np.median(e[:,:-1],axis=0)),'rate_med',np.median(e[:,-1]))
    print(mode,'worst cell medians',np.max([np.max(np.median(e[:,:-1],axis=0)) for e in errs]),np.max([np.median(e[:,-1]) for e in errs]))
out=m.solve_alpha_m97_c03_batch_02(d); print('full tau',out['estimated_taus'],'full weights',out['hereditary_weights']); print('public truth field names',list(item['gt']))
c=d['calibration_load_cells']; base=m._calibration(c); lab_ids=c['lab_id']; clean_keep=(lab_ids!=0)|(np.cumsum(lab_ids==0)==1)
assert np.isfinite(m._calibration({k:v[clean_keep] for k,v in c.items()})).all()
for lab in range(4):
    for spike in (-7,7):
        injected={k:v.copy() for k,v in c.items()}; injected['true_stress'][np.flatnonzero(lab_ids==lab)[-1]]+=spike
        err=np.max(abs(m._calibration(injected)-base)); assert err<1e-4,err
print('sparse calibration and eight EMI interventions pass')
