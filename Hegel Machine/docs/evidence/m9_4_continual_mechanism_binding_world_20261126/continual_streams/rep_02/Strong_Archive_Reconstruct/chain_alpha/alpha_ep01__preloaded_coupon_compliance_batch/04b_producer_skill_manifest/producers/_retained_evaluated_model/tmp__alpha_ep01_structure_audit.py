import importlib.util, pickle
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
p='/holdout_task/alpha_ep01_solver.py'
spec=importlib.util.spec_from_file_location('solver',p)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep01.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
tr,te=d['train'],d['test']; a,b,y,complete=s._prepare(d); c=complete[:len(y)]
t=np.asarray(d['eval_times']); labs=tr['lab_id']; strata=tr['stratum']; den=np.maximum(abs(a.hold),.5)
x=np.concatenate([tr['coupon_features_raw'],te['coupon_features_raw']]).astype(float)
l=np.r_[labs,te['lab_id']]; st=np.r_[strata,te['stratum']]; refs=np.concatenate([tr['ref_coupons'],te['ref_coupons']])
for j in np.unique(l):
    ix=np.flatnonzero(l==j); rr=refs[ix].T
    observed=[col[np.isfinite(col)] for col in rr]; observed=[col for col in observed if col.size]
    offset=np.median([np.mean(col) for col in observed]); gain=np.median([np.std(col, ddof=0) for col in observed]); x[ix]=(x[ix]-offset)/gain
for k in range(x.shape[1]):
    finite=np.isfinite(x[:,k]); means=[np.mean(x[finite & (l==j) & (st==q),k]) for j,q in sorted(set(zip(l[finite],st[finite])))]; x[~finite,k]=np.mean(means)
np.testing.assert_allclose(x,s._aligned_panels(d)[0],atol=1e-12)
print('Independent reference alignment and cell-balanced imputation agree.')
x=x[:len(y)]
pairs=list(combinations(range(x.shape[1]),2)); triples=list(combinations(range(x.shape[1]),3))
# A linear pairwise sum or difference is already spanned by the four main effects.
pol=np.column_stack([x*x]+[x[:,i]*x[:,j] for i,j in pairs]+[x[:,i]*x[:,j]*x[:,k] for i,j,k in triples])
rat=np.column_stack([x/(1+0.5*x*x)]+[(x[:,i]+sign*x[:,j])/(1+0.5*(x[:,i]+sign*x[:,j])**2) for i,j in pairs for sign in (-1,1)])
load=np.column_stack([a.hold**2,a.hold/(1+0.5*a.hold**2),a.hold[:,None]*x])
pre=np.linalg.norm(tr['initial_memory_states'],axis=1)
slices={**{f'lab{j}':labs==j for j in np.unique(labs)},**{f'stratum{j}':strata==j for j in np.unique(strata)},'complete':c,'missing':~c,'low_hold':abs(a.hold)<.5,'high_preload':pre>=np.quantile(pre,.9)}
models={'base':None,'polynomial drift':pol,'rational drift':rat,'stress-coupled drift':load}
errs={k:np.empty_like(y[:,:-1]) for k in models}; dr=np.empty(len(y))
for j in np.unique(labs):
    test=labs==j; fit=~test & c
    tau,coef=s._fit(a,y,fit); pred=a.matrix(np.log(tau))@coef; r=y-pred; dr[test]=r[test,-1]
    # Median over horizons resists the isolated strain acquisition spikes.
    target=np.median(r[:,2:-1]/t[2:],axis=1)
    rate_gain=np.where(c,.25,1.)
    for name,f in models.items():
        shift=np.zeros(len(y))
        if f is not None:
            f=(f-np.mean(f[fit],axis=0))/np.maximum(np.std(f[fit],axis=0),1e-8)
            f=np.column_stack([np.ones(len(y)),f])
            w=least_squares(lambda w: (f[fit]@w-target[fit])/.012,np.zeros(f.shape[1]),loss='soft_l1').x
            shift=f@w
        errs[name][test]=r[test,:-1]-(rate_gain[test]*r[test,-1]+(1-rate_gain[test])*shift[test])[:,None]*t
clean=abs(errs['base'])<.4
for name,e in errs.items():
    def score(m):
        ae=abs(e[m]); ok=clean[m]; sec=ae[:,-1]/den[m]; return [np.sqrt(np.mean(ae**2)),np.sqrt(np.mean(ae[ok]**2)),np.quantile(ae,.9),np.quantile(sec[ok[:,-1]],.9)]
    print(name,'worst [all RMSE, nonspike RMSE, curve p90, nonspike secant p90]',np.round(np.max([score(m) for m in slices.values()],axis=0),5),'complete',np.round(score(c),5))
print('Held-out rate residual median by lab:',[round(float(np.median(dr[(labs==j)&c])),5) for j in np.unique(labs)])
cal=d['calibration_load_cells']
for j in np.unique(labs):
    ix=cal['lab_id']==j; xx=cal['cmd_stress'][ix]; yy=cal['true_stress'][ix]; errs_cal=[]
    for deg in (1,2):
        e=[yy[i]-np.polyval(np.polyfit(np.delete(xx,i),np.delete(yy,i),deg),xx[i]) for i in range(len(xx))]; errs_cal.append(np.sqrt(np.mean(np.square(e))))
    print('Calibration lab',j,'leave-one-point-out RMSE [affine, quadratic]',np.round(errs_cal,6))
