import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m99_rank2 import *
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); q=s.loader.load_module()
Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; missing=np.isnan(g['coupon_features_raw']).any(1); labs=g['lab_id']; st=g['stratum']; ls=np.unique(labs)
def report(name,e):
 print(name,'worst horizon, all rate, complete rate, missing rate, worst lab curve/rate, worst stratum curve/rate:',*[round(float(v),6) for v in (np.sqrt(np.mean(e[:,:-1]**2,axis=0)).max(),np.sqrt(np.mean(e[:,-1]**2)),np.sqrt(np.mean(e[~missing,-1]**2)),np.sqrt(np.mean(e[missing,-1]**2)),*(max(np.sqrt(np.mean(e[ids==k,sl]**2)) for k in np.unique(ids)) for ids in (labs,st) for sl in (slice(None,-1),-1)))])
report('unchanged', E)
for name,Z in [('pooled',G),('chamber-only',q._features(g,g)[:len(labs)])]:
 c=q._coupon_inputs(g,Z,bridge); cu,ra=q._chamber_response(taus,theta,c,dt,t); e=Y-np.c_[cu,ra]; report(name,e); print(name,'missing rate errors',e[missing,-1].round(6))
 B=np.r_[-np.expm1(-t),1.]; score=c[2]@theta[2:]; I=labs[:,None]==ls; print(name,'amplitude residuals by lab',[(int(l),float(np.mean((e@B/(B@B))[(labs==l)&~missing]))) for l in ls]);
 for n,A in [('offset',I),('offset+gain',np.c_[I,I*(score-theta[2])[:,None]])]:
  res=e.copy(); a=e@B/(B@B)
  for j in range(len(e)):
   keep=(np.arange(len(e))!=j)&~missing; res[j]-=(A[j]@np.linalg.lstsq(A[keep],a[keep],rcond=None)[0])*B
  report(name+' LOO '+n,res)
out=q.solve_batch(d); print('all finite', all(np.isfinite(a).all() for a in out.values())); assert all(np.isfinite(a).all() for a in out.values()); assert .05<out['estimated_taus'][0]<out['estimated_taus'][1]<25
