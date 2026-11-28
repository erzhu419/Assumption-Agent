import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import downstream_solver as s
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
a,b,c=(d[k] for k in ('train','test','chamber_verification_trials'))
t,dt=d['eval_times'],d['past_dt'];n=len(a['lab_id']);x=s._features(a,b)
u,v=s._fit(x[:n],*s._loads(a),dt,t,a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs'])
p,h=s._loads(c); z=np.zeros((len(h),2))
for j,step in enumerate(dt):
    gain=-np.expm1(-step/u);z+=(p[:,j,None]-z)*gain
X=s._features(c,c)[:len(h)];D=v[2]+X@v[3:];R=np.column_stack((1/u,-np.expm1(-t/u[:,None])))
Y=np.column_stack((c['instant_rate_obs'],c['strain_curves_obs']-c['strain_0'][:,None]));E=Y-(h[:,None]*v[:2])@R
L=np.r_[1,t];F=u[:,None]*R-L; labs=c['lab_id']; complete=np.isfinite(c['coupon_features_raw']).all(axis=1)
by_model={}
for j in np.unique(labs):
    mask=labs!=j; ii=~mask
    def lin_fit(terms):
        A=np.stack(terms,axis=-1);target=E-D[:,None]*L
        coef=np.linalg.lstsq(A[mask].reshape(-1,A.shape[-1]),target[mask].ravel(),rcond=None)[0]
        return D[:,None]*L+A@coef,coef
    terms=[D[:,None]*f for f in F]
    models={'unchanged':D[:,None]*L-(z*v[:2])@R,'two_branch':lin_fit(terms)[0]}
    models['partial_memory'],mc=lin_fit(terms+[-z[:,k,None]*v[k]*R[k] for k in range(2)])
    contrast=(h-p[:,-1]); contrast=contrast/np.sqrt(np.mean(contrast[mask]**2))
    models['stress_rational'],rc=lin_fit(terms+[D[:,None]*L*(contrast**2/(1+contrast**2))[:,None]])
    for name,kernel in {'log':lambda q:np.log1p(q*t)/q,'sqrt':lambda q:2*t/(1+np.sqrt(1+q*t)),'rational_time':lambda q:t/(1+q*t)}.items():
        def pred(q):return D[:,None]*np.r_[1,kernel(q[0])]
        q=least_squares(lambda q:(pred(q)-E)[mask].ravel(),[.5],bounds=(.0001,20)).x
        models[name]=pred(q)
    print('lab',j,'retained memory',mc[-2:].round(4),'rational contrast interaction',round(rc[-1],4))
    for name,P in models.items():
        out=by_model.setdefault(name,np.empty_like(E));out[ii]=(E-P)[ii]
for name,e in by_model.items():
    rmse=lambda r: (np.sqrt(np.mean(r[:,0]**2)),np.sqrt(np.mean(r[:,1:]**2,axis=0)).max())
    worst=np.max([rmse(e[labs==j]) for j in np.unique(labs)],axis=0)
    print(name,'worst lab (rate, strain)',np.round(worst,6),'complete OOF',np.round(rmse(e[complete]),6),'missing OOF',np.round(rmse(e[~complete]),6))
