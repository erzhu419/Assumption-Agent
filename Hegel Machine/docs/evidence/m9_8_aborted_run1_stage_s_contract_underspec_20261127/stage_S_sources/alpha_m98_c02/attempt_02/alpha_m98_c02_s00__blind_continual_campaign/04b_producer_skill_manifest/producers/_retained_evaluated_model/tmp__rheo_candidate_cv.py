import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
t,dt=d['eval_times'],d['past_dt']; bridge=m._load_bridge(d['calibration_load_cells'])
def p90(e,r,rows):
    return np.round([np.quantile(abs(e[rows]),.9),np.quantile(abs(r[rows]),.9),np.quantile(abs(e[rows,-3:]),.9)],5)
for key in ['train','chamber_verification_trials']:
    g=d[key]; pilot=key!='train'; prep=m._prepare(g,bridge); n=len(prep[0]); labs=g['lab_id']; complete=prep[3]
    y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; fit=m._fit(g,prep,t,dt,pilot=pilot); print(key,'full fit',fit)
    for policy,folds in [('LOLO',[labs==l for l in np.unique(labs)]),('LOO',[np.arange(n)==i for i in range(n)])]:
        E=np.zeros_like(y); R=np.zeros_like(r); taus=[]
        for holdout in folds:
            f,w=m._fit(g,prep,t,dt,pilot=pilot,mask=~holdout); A,B=m._design(prep,t,dt,f)
            E[holdout]=A[holdout]@w-y[holdout]; R[holdout]=B[holdout]@w-r[holdout]; taus.append(f)
        print(policy,'complete/missing',p90(E,R,complete),p90(E,R,~complete),'worst lab',np.max([p90(E,R,labs==l) for l in np.unique(labs)],axis=0),'tau min/max',np.min(taus,axis=0),np.max(taus,axis=0))
