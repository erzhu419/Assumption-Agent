exec(open('/tmp/stress_thermal.py').read().split('for c in cases:')[0])
from scipy.optimize import least_squares
seed=25
rng=np.random.default_rng(seed)
n=22; gamma=5; sigma=.125; xy=[.96,.48]
x,y=(np.indices((n,n))+.5)/n
k=np.exp(.65*np.sin(6*x+2*y)+.6*np.cos(3*x-8*y)); k=np.where(x>.52,3*k,k)
amp=18*rng.uniform(.8,1.5)*(.085/sigma)**2; offset=rng.uniform(-.08,.08)
q,temp=forward(k,gamma,sigma,xy,amp)
m=40; sensor=np.column_stack(np.unravel_index(rng.choice(n*n,m,replace=False),(n,n)))
clean=temp[tuple(sensor.T)]+offset; obs=clean+rng.normal(0,.0006,m)
bad=rng.choice(m,round(.06*m),replace=False); obs[bad]+=rng.uniform(.07,.22,len(bad))*rng.choice([-1,1],len(bad))
measurement=dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=sensor,T_obs=obs)
base=s.localize_thermal_source(measurement)
p0=np.r_[base['source_xy'],base['amplitude'],base['junction_offset']]
truth=np.r_[xy,amp,offset]; g=s._sensor_green(k,gamma,sensor)
def residual(p,target=obs):
    q=np.exp(-((x-p[0])**2+(y-p[1])**2)/(2*sigma**2)); return p[2]*(g@q.ravel())+p[3]-target
bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf]); real=np.setdiff1d(np.arange(m),bad)
r=residual(p0); cut=1.4826*np.median(abs(r-np.median(r)))
print('truth',truth,'baseline',p0,'MAD scale',cut,'bad residuals',r[bad]); print('actual inlier residuals at truth',np.quantile(abs(residual(truth)[real]),[.5,.9,1]))
def report(label, fit):
    p=fit.x; print(label,p, 'signed amp error',p[2]/amp-1,'inlier SSE',np.sum(residual(p)[real]**2))
for a in [1,1.5,2,2.5,3,4,5,8]:
    report(f'Cauchy {a} MAD', least_squares(residual,p0,bounds=bounds,x_scale='jac',loss='cauchy',f_scale=a*cut,gtol=1e-13))
for a in [3,4,5,6,8,12]:
    mask=abs(r-np.median(r))<=a*cut
    report(f'Hardened {a} MAD, kept {mask.sum()}',least_squares(lambda p:residual(p)[mask],p0,bounds=bounds,x_scale='jac',gtol=1e-13))
oracle=least_squares(lambda p:residual(p)[real],p0,bounds=bounds,x_scale='jac',gtol=1e-13)
report('Known inliers',oracle)
print('known-inlier fitted x/amplitude correlation',np.corrcoef(np.linalg.pinv(oracle.jac)[:3])[:3,:3]); print('known-inlier local standard errors',.0006*np.sqrt(np.diag(np.linalg.pinv(oracle.jac.T@oracle.jac))))
