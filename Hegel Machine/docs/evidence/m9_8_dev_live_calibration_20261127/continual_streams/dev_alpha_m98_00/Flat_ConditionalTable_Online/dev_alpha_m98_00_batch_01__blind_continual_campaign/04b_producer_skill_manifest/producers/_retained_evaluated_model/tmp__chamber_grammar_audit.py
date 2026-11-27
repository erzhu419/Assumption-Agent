import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py'); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; q=item['campaign_diagnostics']; out=m['solve_batch'](d); tau=out['estimated_taus']; th=out['hereditary_weights']; t=d['eval_times']; c=m['_stress_calibration'](d['calibration_load_cells']); h=q['hold_channel_raw']@c[:-1]+c[-1]; past=q['past_channel_raw']@c[:-1]+c[-1]; X=m['_aligned_features'](q,q)[len(h):]; D=X@th[2:]; z=np.zeros((len(h),2))
for i,dt in enumerate(d['past_dt']):
    z += (-np.expm1(-dt/tau))*(past[:,i,None]-z)
U=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); J=U*tau; T=np.r_[t,1.0]; B=h[:,None]*(U@th[:2])+D[:,None]*T; V=J-T[:,None]; Y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; M=-z[:,None,:]*U[None,:,:]*th[:2]; scale=np.median(D**2)
def predict(kind, train, rows):
    def correction(p, k):
        drift=D[k]
        if kind=='saturation': drift=drift/(1+p[2]*drift**2/scale)
        v=drift[:,None]*(V@p[:2]);
        if kind=='residual memory': v+=M[k]@p[2:]
        return v
    p=np.array([.52,.43]);
    if kind!='fixed reset':
        lo=[0.,0.]; hi=[1.,1.]
        if kind=='saturation': p=np.r_[p,.01]; lo+=[0.]; hi+=[5.]
        if kind=='residual memory': p=np.r_[p,.01,.01]; lo+=[0.,0.]; hi+=[1.,1.]
        p=least_squares(lambda p:(B[train]+correction(p,train)-Y[train]).ravel(),p,bounds=(lo,hi)).x
    return B[rows]+correction(p,rows)
for kind in ('fixed reset','refitted reset','saturation','residual memory'):
    for group in ('lab_id','stratum'):
        scores=[]
        for label in np.unique(q[group]):
            k=q[group]==label; e=abs(predict(kind,~k,k)-Y[k]); scores.append((np.quantile(e[:,:-1],.9),np.quantile(e[:,-1],.9)))
        print(kind,group,'worst-group P90 strain/rate',np.max(scores,axis=0).round(6),'per-group',np.round(scores,6).tolist())
