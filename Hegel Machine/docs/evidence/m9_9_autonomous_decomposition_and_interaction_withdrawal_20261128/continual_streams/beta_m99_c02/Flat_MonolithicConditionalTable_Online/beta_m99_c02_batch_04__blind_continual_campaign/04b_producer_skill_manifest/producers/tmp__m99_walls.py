import importlib.util, itertools, json, pickle, numpy as np

def load(tag):
    path=f'/tmp/hegel_continual_archive/beta_m99_c02_{tag}_solver.py'
    spec=importlib.util.spec_from_file_location(tag,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=load('batch_03'); periodic=load('batch_02'); y=a._calibrate(s)
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for entry, following in zip(log, ['s00','batch_01','batch_02','batch_03','current']):
    means=np.array(entry['facility_telemetry_trace']).mean(0)
    print('after',entry['completed_campaign_short_name'],'before',following,'state means',means[[1,7]])

def axis(k,mode):
    kap,vel,h=s['kappa_'+k],s['vel_'+k],s['L'+k]/(s['n'+k]-1)
    return periodic._axis(kap,vel,h) if mode=='P' else a._axis(kap,vel,h,s['robin_'+k] if mode=='O' else 0.)

def fit(S,y,train):
    sub={**s,'Z_nuisance':s['Z_nuisance'][train]}
    x,r=a._profile(S[train],y[train],sub)
    w,nu=a._box_simplex(x.T@x+s['source_ridge']*np.eye(S.shape[1]),-x.T@r,s['u_box'],s['total_budget'])
    z=sub['Z_nuisance']; gamma=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@(y[train]-S[train]@w))
    return w,nu,x@w-r,gamma

n=len(y); shifts=np.repeat(np.arange(len(s['mu_shifts'])),len(s['sensor_ij'])); bank=s['row_bank_id']; ij=np.tile(s['sensor_ij'],(len(s['mu_shifts']),1)); edge=(ij[:,0]<2)|(ij[:,0]>=s['nx']-2)|(ij[:,1]<2)|(ij[:,1]>=s['ny']-2)
for modes in itertools.product('ORP',repeat=2):
    S,U=a._responses(s,[axis(k,m) for k,m in zip('xy',modes)])
    w,nu,e,gamma=fit(S,y,np.ones(n,bool))
    pred=[]
    for validation in [shifts==i for i in range(3)]+[bank==i for i in range(4)]+[edge,~edge]:
        w1,_,_,g1=fit(S,y,~validation)
        d=S[validation]@w1+s['Z_nuisance'][validation]@g1-y[validation]
        pred.append(np.sqrt(np.mean(d*d)))
    print(''.join(modes),'pooled %.6f'%np.sqrt(np.mean(e*e)), 'CV shift/bank/edge',np.round(pred,6),'weights',np.round(w,4))
