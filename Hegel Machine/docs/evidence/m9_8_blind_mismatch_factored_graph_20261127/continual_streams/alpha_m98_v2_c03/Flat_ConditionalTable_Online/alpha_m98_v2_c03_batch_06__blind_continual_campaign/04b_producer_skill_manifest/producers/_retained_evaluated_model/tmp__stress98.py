import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('public items:',len(items))
i=items[0]; d=i['dataset']; g=i['campaign_diagnostics']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; good=np.max(abs(y-np.median(y,axis=0)),axis=1)<8
# Hold out all nominal single-axis rows together, or high-norm mixed loads together.
axis=(np.count_nonzero(x,axis=1)==1); high=np.linalg.norm(x,axis=1)>=np.quantile(np.linalg.norm(x[~axis],axis=1),.8)
for name,test in [('axis',axis),('strong_mixed',high&~axis)]:
    fit=good&~test; test=good&test
    model=m._calibration(dict(true_channel_loads=x[fit],bridge_raw=y[fit])); e=(m._loads(y[test],model)-x[test]).sum(1)
    print(name,'inverse total-load p90/max',len(e),np.quantile(abs(e),.9),max(abs(e)))
    z=np.arange(4); o,L,q,b=model; X=x[test]; R=y[test]-(X@L+o+q*X*X[:,m._NEIGHBOR]+b*X**3)
    # A missing pairwise difference/sum or rational saturation would leave a coherent residual.
    X=x[fit]; Y=y[fit]; T=x[test]; yh=np.empty_like(R); yc=np.empty_like(R)
    for j,k in enumerate(m._NEIGHBOR):
        def terms(v): return np.c_[np.ones(len(v)),v,v[:,j]*v[:,k],v[:,j]**3]
        a=terms(X); at=terms(T); p=np.linalg.lstsq(a,Y[:,j],rcond=None)[0]; yc[:,j]=at@p
        candidates=[]
        for sign in (-1,1):
            def rat(v): return (v[:,j]+sign*v[:,k])**2
            s=rat(X); st=rat(T); groups=np.array_split(np.random.default_rng(27).permutation(len(a)),4)
            for h in (0.,.003,.01,.03,.1,.3):
                A=a/(1+h*s[:,None]); AT=at/(1+h*st[:,None]); err=[]
                for valid in groups:
                    tr=~np.isin(np.arange(len(a)),valid); beta=np.linalg.lstsq(A[tr],Y[tr,j],rcond=None)[0]; err.extend((A[valid]@beta-Y[valid,j]).tolist())
                candidates.append((np.mean(np.square(err)),h,AT@np.linalg.lstsq(A,Y[:,j],rcond=None)[0]))
        _,_,yh[:,j]=min(candidates,key=lambda r:r[0])
    print(name,'cubic vs nested rational channel p90',*[float(np.quantile(abs(p-y[test]),.9)) for p in (yc,yh)])
tr,te=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; train=d['train']; target=lambda p:np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]; tau,theta=m._fit(tr,dt,t,target(train)); cp=m._coupon(g,m._features(g,g)[-len(g['lab_id']):],m._calibration(c)); strain,rate=m._chamber_predict(cp,t,tau,theta); e=np.c_[strain,rate]-target(g)
for lab in np.unique(g['lab_id']):
    ix=g['lab_id']==lab; group=e[ix]; print('chamber lab',lab,'signed medians',np.median(group[:,:-1]),np.median(group[:,-1]),'p90',np.quantile(abs(group[:,:-1]),.9),np.quantile(abs(group[:,-1]),.9))
for mask in (np.any(np.isnan(g['coupon_features_raw']),axis=1),np.all(np.isfinite(g['coupon_features_raw']),axis=1)):
    if mask.any(): print('missingness slice',mask.sum(),'strain/rate p90',np.quantile(abs(e[mask,:-1]),.9),np.quantile(abs(e[mask,-1]),.9))
for ds in (item['dataset'] for item in items):
    out=m.solve_batch(ds); assert all(np.isfinite(a).all() for a in out.values())
