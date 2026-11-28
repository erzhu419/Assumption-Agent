exec(open('/tmp/chamber_cv.py').read().split('for label,X')[0])
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation as mad

def estimate(tau,theta):
    basis, r0 = m._basis(tau,t); b=np.vstack((basis,r0)); z=np.zeros((len(h),2))
    for dur,s in zip(dt,p.T): z+=-np.expm1(-dur/tau)*(s[:,None]-z)
    mech=((h[:,None]-z)*theta[:2])@b[:,:2].T
    redistribution=b[:,:2]*tau-b[:,2,None]; latent=y[:,-1]-mech[:,-1]
    for kind,force,sel in [('measured rate',latent,np.ones(len(h),bool)),('predicted force, complete',m._features(q)[0]@theta[2:],~miss),('predicted force, all',m._features(q)[0]@theta[2:],np.ones(len(h),bool))]:
      P=force[:,None,None]*redistribution[None,:,:]; R=y-mech-force[:,None]*b[:,2]
      g=np.linalg.lstsq(P[sel].reshape(-1,2),R[sel].ravel(),rcond=None)[0]
      print(kind, np.round(g,6), 'RMSE', score((P@g-R)[sel]))

print('staged params',tau,theta)
estimate(tau,theta)
for name,payload in [('staged',pay),('dynamic',None)]:
 d2=calibrate_upstream(raw,payload); train=d2['train']; hist=train['past_channel_raw'].sum(-1); hold=train['hold_channel_raw'].sum(-1)
 ob=np.r_[(train['strain_curves_obs']-train['strain_0'][:,None]).ravel(), train['instant_rate_obs']]
 def e(v): return m._design(np.exp(v[:2]),hist,hold,x,dt,t)@v[2:]-ob
 v0=np.r_[np.log(tau),theta]; eps=e(v0); n=len(hold); print(name,'raw difference',np.max(abs(d2['test']['hold_channel_raw']-te['hold_channel_raw'])), 'noise MAD',mad(eps[:-n],scale='normal'),mad(eps[-n:],scale='normal')); print('rate quantiles',np.quantile(abs(eps[-n:]),[.5,.8,.95,.99,1]))
 for factor in [1,2,4]:
    scale=np.r_[np.full(len(eps)-n,.009),np.full(n,mad(eps[-n:],scale='normal'))]; v=least_squares(lambda v:e(v)/scale,v0,loss='cauchy',f_scale=factor).x
    print('factor',factor,np.exp(v[:2]),v[2:],'mad',mad(e(v)[:-n],scale='normal'),mad(e(v)[-n:],scale='normal'))
    if name=='staged': estimate(np.exp(v[:2]),v[2:])
