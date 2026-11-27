from m98_probe import *
from scipy.optimize import least_squares
O=obs(q); B,R=m._branches(c,dt,t,tau); Y=O[:,:-1]-B@theta[:2]; drift=c[2]@theta[2:]; rate_drift=O[:,-1]-R@theta[:2]; complete=~np.isnan(q['coupon_features_raw']).any(1); labs=q['lab_id'];
models={
 'linear':([],lambda p,u:u),
 'integrated_exp':([1],lambda p,u:-np.expm1(-p[0]*u)/p[0]),
 'rational_t':([.75],lambda p,u:u/(1+p[0]*u)),
 'rational_t2':([.2],lambda p,u:u/(1+p[0]*u**2)),
 'time_exp':([.35],lambda p,u:u*np.exp(-p[0]*u)),
 'exp_floor':([1,.1],lambda p,u:p[1]*u+(1-p[1])*(-np.expm1(-p[0]*u)/p[0])),
 'time_exp_floor':([.6,.2],lambda p,u:u*(p[1]+(1-p[1])*np.exp(-p[0]*u))),
 'rational_floor':([.8,.1],lambda p,u:u*(p[1]+(1-p[1])/(1+p[0]*u))),
 'stretched_rational':([.8,1.1],lambda p,u:u/(1+p[0]*u**p[1])),
 'stretched_exp':([.35,1.1],lambda p,u:u*np.exp(-p[0]*u**p[1])),
 'rational_quadratic':([.7,.1],lambda p,u:u/(1+p[0]*u+p[1]*u**2)),
}
for name,(seed,fun) in models.items():
 def run(ix):
  return least_squares(lambda p:(Y[ix]-drift[ix,None]*fun(p,t)).ravel(),seed,bounds=(0,np.inf),loss='soft_l1',f_scale=.005).x if seed else []
 p=run(complete); e=Y-drift[:,None]*fun(p,t); cv=[abs(Y[complete&(labs==l)]-drift[complete&(labs==l),None]*fun(run(complete&(labs!=l)),t)).mean() for l in np.unique(labs)]; free=Y@fun(p,t)/np.sum(fun(p,t)**2); print(name,np.round(p,6),'complete mae',round(abs(e[complete]).mean(),5),'worst LOLO',round(max(cv),5),'worst-horizon bias',round(max(abs(e[complete].mean(0))),5),'free-amplitude mae',round(abs(Y-free[:,None]*fun(p,t)).mean(),5))
print('empirical effective time from complete coupons',np.c_[t,(drift[complete]@Y[complete])/(drift[complete]@drift[complete]),(rate_drift[complete]@Y[complete])/(rate_drift[complete]@rate_drift[complete])].round(6));
