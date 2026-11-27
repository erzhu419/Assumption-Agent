import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import alpha_m97_c01_batch_03_solver as s
with open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
dt,t,past,hold,x=s._prepare(d); tr=d['train']; n=len(tr['lab_id'])
past,hold,x=past[:n],hold[:n],x[:n]
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']
complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)

def fit(m,loss):
    tau,w=s._fit(dt,t,past[m],hold[m],x[m],yc[m],yr[m]); y=np.r_[yc[m].ravel(),yr[m]]
    def residual(p):
        a,b=np.exp(p[:2]); c,r=s._basis(np.array([a,a+b]),dt,t,past[m],hold[m],x[m]); return np.r_[(c@p[2:]).ravel(),r@p[2:]]-y
    p=np.r_[np.log([tau[0],tau[1]-tau[0]]),w]; err=residual(p); k=yc[m].size
    def mad(e): return max(1e-8,1.4826*np.median(abs(e-np.median(e))))
    scale=np.r_[np.repeat(mad(err[:k]),k),np.repeat(mad(err[k:]),m.sum())]
    if loss=='linear': return tau,w,scale[[0,-1]]
    for _ in range(3):
        opt=least_squares(lambda p:residual(p)/scale,p,loss=loss,f_scale=2,max_nfev=120)
        p=opt.x; err=residual(p); scale[:k]=mad(err[:k]); scale[k:]=mad(err[k:])
    a,b=np.exp(p[:2]); return np.array([a,a+b]),p[2:],scale[[0,-1]]

for mname,base in [('all',np.ones(n,dtype=bool)),('complete',complete)]:
    for loss in ('linear','soft_l1','cauchy'):
        if mname=='complete' and loss=='linear': continue
        tau,w,scale=fit(base,loss); print(mname,loss,'full taus',tau,'weights',w.round(5),'curve/rate MAD scales',scale)
        metrics=[]
        for c in range(4):
            test=tr['lab_id']==c; train=~test & base
            tau,w,scale=fit(train,loss); a,b=s._basis(tau,dt,t,past[test],hold[test],x[test]); ec,er=a@w-yc[test],b@w-yr[test]
            q=np.quantile(np.abs(ec),[.5,.90,.95]); qc=np.quantile(np.abs(ec[complete[test]]),[.5,.90,.95]); qr=np.quantile(np.abs(er),[.5,.90,.95]); qrc=np.quantile(np.abs(er[complete[test]]),[.5,.90,.95]); metrics.append(np.r_[q,qc,qr,qrc]); print('  lab',c,'tau',tau.round(4),'curve all/complete and rate all/complete median/p90/p95',metrics[-1].round(5))
        print('  WORST',np.max(metrics,axis=0).round(5))
