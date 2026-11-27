import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr,te=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; target=lambda p:np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]; y=target(d['train']); tau,th=m._fit(tr,dt,t,y)
gp=m._coupon(g,m._features(g,g)[-len(g['lab_id']):],m._calibration(d['calibration_load_cells'])); ex=np.r_[-np.expm1(-t[:,None]/tau),[1/tau]]; base=gp[1][:,None]*(ex@th[:2]); A=(gp[2]@th[2:])[:,None,None]*ex; Y=target(g)-base; sc=np.r_[np.repeat(.01158514,len(t)),.00848624]
for heldout in [np.arange(len(Y)),g['lab_id']]:
 for name,q in [('fitted',None),('unit initial',1),('40/60',np.array([.4,.6])*tau),('archived',np.array([.19694,2.09244]))]:
    E=np.empty_like(Y)
    for j in np.unique(heldout):
        test=heldout==j; a=(A[~test]/sc[:,None]).reshape(-1,2); b=(Y[~test]/sc).ravel();
        if q is None: c=least_squares(lambda p:a@p-b,[.2,2.],loss='soft_l1').x
        elif isinstance(q,int): c=least_squares(lambda p:a@(tau*np.r_[p,1-p])-b,[.4],loss='soft_l1').x; c=tau*np.r_[c,1-c]
        else: c=q
        E[test]=A[test]@c-Y[test]
    print('diagnostic',len(np.unique(heldout)),name,'curve/rate p90',np.round([np.quantile(abs(E[:,:-1]),.9),np.quantile(abs(E[:,-1]),.9)],6),'worst lab curve/rate median',np.max([np.median(abs(E[g['lab_id']==j]),axis=0)[[-2,-1]] for j in np.unique(g['lab_id'])],axis=0))
E=np.zeros_like(y)
for group in np.unique(d['train']['lab_id']):
    ix=d['train']['lab_id']==group; a,b=m._fit(tuple(v[~ix] for v in tr),dt,t,y[~ix]); E[ix]=m._design(tuple(v[ix] for v in tr),dt,t,a)@b-y[ix]; print('benchtop held-out lab',group,'taus',a,'strain/rate p90',np.round([np.quantile(abs(E[ix,:-1]),.90),np.quantile(abs(E[ix,-1]),.90)],6))
print('final full fit',tau,th)
