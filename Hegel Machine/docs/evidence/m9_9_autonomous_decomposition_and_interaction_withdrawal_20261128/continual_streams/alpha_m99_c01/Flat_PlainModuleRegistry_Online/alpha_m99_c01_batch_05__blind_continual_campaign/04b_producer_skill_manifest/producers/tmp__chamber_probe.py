import sys,pickle,importlib.util
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=i['dataset']; d=calibrate_upstream(dict(raw,chamber_verification_trials=i['campaign_diagnostics']))
a,b,c=(d[k] for k in ('train','test','chamber_verification_trials')); dt,t=d['past_dt'],d['eval_times']; x,z=m._features(a,b); _,q=m._features(a,c)
loads=lambda p:tuple(p[k].sum(axis=-1) for k in ('past_channel_raw','hold_channel_raw'))
p,h=loads(a); y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; tau,theta=m._fit(x,p,h,dt,t,y,r)
print('material',tau,theta,'times',t,'dt',dt)
P,H=loads(c); Y=c['strain_curves_obs']-c['strain_0'][:,None]; R=c['instant_rate_obs']; lab=c['lab_id']; print('diagnostic labels',np.bincount(lab)); print('loads',np.c_[H,P][:4]);print('curves and rate',np.c_[R[:4],Y[:4]])
A,B=m._design(tau,q,P,H,dt,t); print('benchtop rmse',np.sqrt(np.mean((A@theta-Y)**2)),np.sqrt(np.mean((B@theta-R)**2)))
def project(u,mask,order):
    T=np.exp(u); F=np.c_[-np.expm1(-t[:,None]/T),*[t**j for j in range(1,order+1)]]; Fr=np.r_[1/T,1,np.zeros(order-1)]; D=np.vstack((F,Fr)); vals=np.c_[Y,R]; coeff=np.linalg.lstsq(D,vals.T,rcond=1e-11)[0].T
    return (coeff@D.T-vals)[mask].ravel(),coeff
for order in (1,2,3):
 for nt in (1,2,3):
    mask=np.ones(len(Y),bool); opts=[least_squares(lambda u:project(u,mask,order)[0],np.log(v),bounds=(np.log(.02),np.log(70)),max_nfev=100) for v in ([.22],[.8],[2.]) if nt==1] if nt==1 else [least_squares(lambda u:project(u,mask,order)[0],np.log(v[:nt]),bounds=(np.log(.02),np.log(70)),max_nfev=100) for v in ([.18,1.5,4.],[.4,2.,12.],[.1,.6,3.])]; opt=min(opts,key=lambda o:o.cost); e,coef=project(opt.x,mask,order);print('empirical',nt,order,np.exp(opt.x),np.quantile(abs(e),[.5,.9,.99,1]),'coeff',coef[:2])
