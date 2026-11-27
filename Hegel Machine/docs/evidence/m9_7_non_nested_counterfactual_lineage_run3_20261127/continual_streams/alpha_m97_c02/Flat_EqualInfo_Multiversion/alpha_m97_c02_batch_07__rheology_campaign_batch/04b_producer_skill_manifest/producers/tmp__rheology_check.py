import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c02_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(p+'public_alpha_m97_c02_batch_07.pkl','rb'))[0]; d=item['dataset']; g=item['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); v=m._prepare(d); t=d['eval_times']; dt=d['past_dt']; C=tr['strain_curves_obs']-tr['strain_0'][:,None]; R=tr['instant_rate_obs']; design=m._design

def stats(x):
    return np.round([np.median(abs(x)),np.quantile(abs(x),.95),np.sqrt(np.mean(x*x))],5)

def log_for_network(past, hold, drift, act, plateau, dt, times, tau):
    return design(past,hold,drift,act,np.ones_like(plateau),dt,times,tau)

for name, fn in [('linear_at_zero_damage',design),('log_at_zero_damage',log_for_network)]:
    print(name,flush=True); m._design=fn
    residual=np.empty_like(C); rate=np.empty_like(R)
    for lab in range(4):
        use=tr['lab_id']!=lab; x=[col[:n] for col in v]
        tau,w=m._fit(tuple(col[use] for col in x),dt,t,C[use],R[use]); a,b=fn(*(col[~use] for col in x),dt,t,tau)
        residual[~use]=a@w-C[~use]; rate[~use]=b@w-R[~use]
        print('LOLO fold',lab,'tau',np.round(tau,4),'strain late by activation',[stats(residual[(~use)&(v[-1][:n]==k),-3:]).tolist() for k in (False,True)],'rate',stats(rate[~use]))
    out=m.solve_alpha_m97_c02_batch_07(d); e=out['predicted_strain']-g['te_clean']; q=out['initial_rate_pred']-g['te_true_rate0'];
    print('full fit',out['estimated_taus'],out['hereditary_weights'],out['regime_mode_weights']); print('public error overall curve/rate',stats(e),stats(q));
    for lab in range(4):
        for mode in (False,True):
            mask=(te['lab_id']==lab)&(v[-1][n:]==mode)
            print('public',lab,mode,'strain',stats(e[mask]),'late signed median',round(float(np.median(e[mask,-3:])),5),'rate',stats(q[mask]))
