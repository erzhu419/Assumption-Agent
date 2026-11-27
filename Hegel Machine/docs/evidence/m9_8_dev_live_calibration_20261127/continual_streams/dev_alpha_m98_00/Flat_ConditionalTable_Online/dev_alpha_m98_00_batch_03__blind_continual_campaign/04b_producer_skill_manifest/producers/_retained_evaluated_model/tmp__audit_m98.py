import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/chamber_probe.py').read())
from scipy.optimize import least_squares
import importlib.util
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
keys=('lab_id','stratum','ref_coupons','coupon_features_raw'); h=len(hold)//2
X=m._aligned_features(*[{k:q[k][j] for k in keys} for j in (slice(None,h),slice(h,None))]); dr=X@theta[2:]; C,R=m._design(tau,dt,t,(past,hold,X)); E=y-C@theta; D=-np.expm1(-t[:,None]/tau)*tau-t[:,None]
def rms(e):return np.sqrt(np.mean(e**2))
def evaluate(name, coord=None):
    fold=[]; coeff=[]
    for lab in np.unique(q['lab_id']):
        val=q['lab_id']==lab; trn=~val
        normal=np.sqrt(np.mean(coord[trn]**2)) if coord is not None else 1
        def predict(p):
            f=dr if coord is None else dr/(1+p[2]*(coord/normal)**2)
            return f[:,None]*(D@p[:2])
        p=least_squares(lambda p:(predict(p)-E)[trn].ravel(), [.63,.61] if coord is None else [.63,.61,0], loss='soft_l1',f_scale=.02,bounds=([-np.inf]*2,[np.inf]*2) if coord is None else ([-np.inf,-np.inf,0],[np.inf,np.inf,np.inf])).x
        fold.append(rms((predict(p)-E)[val])); coeff.append(p)
    print(name,'lab RMSE',np.round(fold,6),'worst',round(max(fold),6),'extra coeff',np.round(np.array(coeff)[:,2:],5).ravel())
evaluate('two fractions'); evaluate('saturation by drift',dr); evaluate('saturation by branch-memory contrast',z[:,0]-z[:,1]); evaluate('saturation by mean branch gap',hold-z.mean(axis=1))
P,V=m._chamber(tau,theta,dt,t,(past,hold,X)); e=P-y; er=V-v
for field in ('lab_id','stratum'):
    for g in np.unique(q[field]):
        a=q[field]==g; print(field,int(g),'n',int(a.sum()),'mean errors strain/rate',round(e[a].mean(),6),round(er[a].mean(),6),'RMSE',round(rms(e[a]),6),round(rms(er[a]),6))
