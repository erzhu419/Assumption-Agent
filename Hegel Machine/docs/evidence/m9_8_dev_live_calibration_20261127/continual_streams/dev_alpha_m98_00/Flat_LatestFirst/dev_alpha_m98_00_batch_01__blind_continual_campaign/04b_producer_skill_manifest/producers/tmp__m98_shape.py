exec(open('/tmp/m98_inspect.py').read().split('for g,x1')[0])
from scipy.optimize import least_squares
x1=x[-len(diag['lab_id']):]; g=diag
p,h=g['past_channel_raw']@cal[:-1]+cal[-1],g['hold_channel_raw']@cal[:-1]+cal[-1]
y=np.column_stack((g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']))
def basis(ts): return np.vstack((np.column_stack((-np.expm1(-t[:,None]/ts),t)),np.r_[1/ts,1]))
def fit(ts, yy=y):
 G=basis(ts); a=np.linalg.lstsq(G,yy.T,rcond=None)[0].T; return a, yy-a@G.T
for st in (tau,[.15,1.4],[.45,5],[.8,12]):
 o=least_squares(lambda v:fit(np.exp(v))[1].ravel(),np.log(st),bounds=(np.log(.03),np.log(200))); print('shape',np.sort(np.exp(o.x)), 'rmse',np.mean(o.fun**2)**.5)
ct=np.sort(np.exp(o.x)); a,e=fit(ct); A,B=m._design(tau,dt,t,(p,h,x1)); base,er=fit(tau); print('shared chamber rmse',np.mean(er**2)**.5, 'baseline curve/rate rmse',np.mean((A@w-y[:,:-1])**2)**.5,np.mean((B@w-y[:,-1])**2)**.5)
for ts,aa in [(tau,base),(ct,a)]:
 z=np.zeros((len(h),2))
 for al,ps in zip(-np.expm1(-dt[:,None]/ts),p.T):z+=al*(ps[:,None]-z)
 print('times',ts,'target amplitude/drift std',aa.std(0)); drift=x1@w[2:]
 for name,F in [('hold+z+drift',np.c_[h,z,drift]),('hold+last+mean+drift',np.c_[h,p[:,-1],p.mean(1),drift]),('hold+all past+drift',np.c_[h,p,drift])]:
  F=np.c_[np.ones(len(h)),F]; cf=np.linalg.lstsq(F,aa,rcond=None)[0]; print(name,'err',np.round(np.mean((F@cf-aa)**2,0)**.5,5),'coef',np.round(cf.T,3).tolist())
