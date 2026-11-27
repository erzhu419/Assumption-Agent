exec(open('/tmp/check_alpha_batch.py').read().split("run('public', d)")[0])
tr,te=d['train'],d['test']; n=len(tr['lab_id']); x, complete=m._features(tr,te)
ga,bi,_=m._calibrate(d['calibration_load_cells']); dt,t=map(np.asarray,(d['past_dt'],d['eval_times']))
j=np.asarray(tr['lab_id']); h=tr['stress_hold_cmd']*ga[j]+bi[j]; past=tr['past_stress_cmd']*ga[j,None]+bi[j,None]; z=tr['initial_memory_states']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; t2=t@t
f=x[:n,1:]; excess=np.maximum(np.abs(h)-float(np.asarray(d['yield_stress']).ravel()[0]),0)*np.sign(h)
terms=np.column_stack([excess,excess**2*np.sign(excess),h/(1+h*h),h/(1+h*h)*f[:,0],*[f[:,a]*f[:,b] for a in range(4) for b in range(a,4)]])
print('yield stress and active rows by lab:', d['yield_stress'],[int(((j==k)&(excess!=0)).sum()) for k in range(4)])
print('public reference vp_theta:', gt.get('vp_theta'))
from scipy.optimize import least_squares
scores=[]
for k in range(4):
  train=complete[:n]&(j!=k); test=complete[:n]&(j==k)
  tau,theta=m._fit(past[train],h[train],z[train],dt,t,x[:n][train],y[train],rate[train]); A,R=m._design(tau,past,h,z,dt,t,x[:n]); e=y-A@theta; er=rate-R@theta
  # Project residual onto the linear-in-time direction using the finite curve samples.
  good=np.isfinite(e); denom=good@(t*t); slope=np.divide(np.where(good,e,0.)@t,denom,out=np.full(n,np.nan),where=denom>0)
  a=train&np.isfinite(slope); b=test&np.isfinite(slope)
  rms=lambda v: np.sqrt(np.mean(v*v))
  scores_k=[]
  for cols in (slice(0,2),slice(2,4),slice(4,None)):
    zc=terms[:,cols]; scale=np.maximum(np.linalg.norm(zc[a],axis=0),1e-12); q=zc/scale
    fit=least_squares(lambda beta:np.r_[q[a]@beta-slope[a],.1*beta], np.zeros(q.shape[1]),loss='soft_l1', f_scale=.018)
    scores_k.append(round(float(rms(slope[b]-q[b]@fit.x)),5))
  print('held lab',k,'finite labels',int(b.sum()),'curve slope rms',round(float(rms(slope[b])),5),'yield / rational / quadratic challenger',scores_k,'median slope',round(float(np.median(slope[b])),5),'median rate residual',round(float(np.nanmedian(er[b])),5))
# Check exact hereditary propagation against the literal step recurrence.
tau=np.array([.37,4.2]); mem=z.copy()
for k,dtk in enumerate(dt):
  mem=past[:,k,None]+(mem-past[:,k,None])*np.exp(-dtk/tau)
_,R=m._design(tau,past,h,z,dt,t,x[:n]); mismatch=np.max(abs(R[:,:2]-(h[:,None]-mem)/tau)); print('literal recurrence max discrepancy',mismatch); assert mismatch<1e-12
