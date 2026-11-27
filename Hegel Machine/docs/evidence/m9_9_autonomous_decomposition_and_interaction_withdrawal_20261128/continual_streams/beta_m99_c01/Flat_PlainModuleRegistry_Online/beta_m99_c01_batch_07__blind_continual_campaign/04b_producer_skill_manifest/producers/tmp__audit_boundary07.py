import contextlib, io, json, runpy, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/holdout_task')
with contextlib.redirect_stdout(io.StringIO()):
    d = runpy.run_path('/tmp/compare_walls07.py')
s,y,z,w0 = (d[k] for k in ('s','y','z','w0'))
m=len(w0); cap=s['u_box']; budget=s['total_budget']; groups=d['groups']; shift=np.repeat(np.arange(len(s['mu_shifts'])),len(d['ix']))
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
    a=np.asarray(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],np.round(np.median(a,axis=0),2))
R=[d['axis'](a,'R') for a in 'xy']; P=[d['axis'](a,'P') for a in 'xy']; O=[d['axis'](a,'O') for a in 'xy']
eq=np.r_[np.ones(m),np.zeros(z.shape[1])]; p0=np.r_[w0,np.zeros(z.shape[1])]
constraints={'type':'eq','fun':lambda p:eq@p-budget,'jac':lambda p:eq}
bounds=[(0,cap)]*m+[(None,None)]*z.shape[1]
def residual(X,t):
    T=X[t]; b=y[t]
    opt=minimize(lambda p:0.5*np.sum((T@p-b)**2),p0,jac=lambda p:T.T@(T@p-b),method='SLSQP',bounds=bounds,constraints=constraints,options={'ftol':1e-13,'maxiter':300})
    assert opt.success,opt.message
    return X@opt.x-y
rms=lambda x:np.sqrt(np.mean(x*x))
# Local perturbations ask whether the supported periodic-x/reflecting-y
# stencil is missing wall leakage or a partial wrap, not whether extra
# parameters can improve the in-sample objective.
a0,b0=P[0],R[1]
for tag,a,b in [('PR',a0,b0),('half_x_wrap',(R[0]+P[0])/2,b0),('quarter_y_wrap',a0,b0+.25*(P[1]-b0)),('2pct_x_leak',a0+.02*(O[0]-R[0]),b0),('2pct_y_leak',a0,b0+.02*(O[1]-R[1]))]:
    X=np.column_stack((d['forward'](a,b),z))
    gg=[rms(residual(X,groups!=k)[groups==k]) for k in np.unique(groups)]
    ss=[rms(residual(X,shift!=k)[shift==k]) for k in np.unique(shift)]
    print(tag,'full',round(rms(residual(X,np.ones(len(y),bool))),7),'rosette aggregate/worst',np.round([rms(np.asarray(gg)),max(gg)],7),'shift aggregate/worst',np.round([rms(np.asarray(ss)),max(ss)],7))
