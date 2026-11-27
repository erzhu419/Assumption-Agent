import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares, minimize
spec=importlib.util.spec_from_file_location('b','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=b._calibrator(s['sensor_calibration_sweeps']); ch=b._Chamber(s); y=c(s['y_obs_raw']); p=s['reference_tracer_pulses']; yref=np.column_stack([c(a['y_pulse_raw']) for a in p]); rhs=ch.F@np.column_stack([a['w_pulse'] for a in p]); ns=len(ch.sensors); r=np.arange(len(y))//ns
rms=lambda z:np.sqrt(np.mean(z*z))
for j in range(len(ch.shifts)):
 keep=r!=j
 def f(w): return (ch.response(w,rhs)[0]-yref)[keep].ravel()
 def jac(w): return ch.response(w,rhs,True)[2][keep].reshape(-1,8)
 fit=least_squares(f,ch.prior,jac=jac,gtol=2e-14)
 print('held shift',j,'rms',rms((ch.response(fit.x,rhs)[0]-yref)[~keep]))
W=b._fit_walls(ch,s,c); S,U,_=ch.response(W); blind=b._fit_walls(ch,dict(s,reference_tracer_pulses=None),c); print('blind walls',blind,'wall prediction max-shift RMS',max(rms(v) for v in np.split(ch.response(blind,rhs)[0]-yref,len(ch.shifts))))
wp=b.solve_batch(s); wb=b.solve_batch(dict(s,reference_tracer_pulses=None)); print('public pulse-free relative errors',np.linalg.norm(wp['weights']-wb['weights'])/np.linalg.norm(wp['weights']),np.linalg.norm(wp['terminal_field']-wb['terminal_field'])/np.linalg.norm(wp['terminal_field']))
Z=s['Z_nuisance']; proj=b._projector(Z,s['nuisance_ridge']); X,yy=proj(S),proj(y); n=X.shape[1]; ridge=s['source_ridge']; H=X.T@X+ridge*np.eye(n); g=-X.T@yy; u=np.broadcast_to(s['u_box'],n); q=wp['kkt_certificate']; w=wp['weights']; print('public equality/stationarity/gap/field consistency',abs(w.sum()-s['total_budget']),np.max(np.abs(H@w+g+q['nu_sum']-q['lam_lower']+q['lam_upper'])),q['primal_objective']-q['dual_lower_bound'],np.max(abs(wp['terminal_field'].ravel(order='F')-U@w)))
rng=np.random.default_rng(153); worst=np.zeros(4)
for i in range(150):
 n=int(rng.integers(3,15)); A=rng.normal(size=(n,n)); H=A.T@A+.04*np.eye(n); g=rng.normal(size=n)*8; u=rng.uniform(.04,.9,n); u[rng.random(n)<.15]=0; B=u.sum()*rng.choice([0.,.01,.5,.99,1.]) if i%2==0 else rng.uniform(.02,.98)*u.sum()
 w,nu=b._box_qp(H,g,u,B); d=H@w+g+nu
 q=lambda t:.5*t@H@t+g@t
 res=minimize(q,u*(B/u.sum()),jac=lambda t:H@t+g,bounds=list(zip(np.zeros(n),u)),constraints=[{'type':'eq','fun':lambda t:t.sum()-B,'jac':lambda t:np.ones(n)}],method='SLSQP',options={'ftol':1e-12})
 low=(w<=1e-9)&(u>0); high=(w>=u-1e-9)&(u>0); mid=~(low|high)&(u>0)
 worst=np.maximum(worst,[abs(w.sum()-B),max(np.max(-w),np.max(w-u)),max(np.max(-d[low],initial=0),np.max(d[high],initial=0),np.max(abs(d[mid]),initial=0)),abs(q(w)-q(res.x)) if res.success else 0])
print('stress worst equality/box/stationarity/oracle objective',worst)
assert np.max(worst[:3])<1e-8
# Compare the empirical inverse on held-range observations under known response laws.
cal=s['sensor_calibration_sweeps']; a=cal['ref_rosette']; test=rng.uniform(-.42,.60,(200,4)); mix=np.array([[1.1,.08,0,.03],[-.03,.91,.05,0],[0,-.04,1.03,.02],[.02,0,.04,.96]])
for name,fn in [('affine',lambda x:x@mix+.04),('cubic-cross',lambda x:(x+.08*x**3)@mix+.09*np.roll(x,1,axis=1)*x+.04),('rational',lambda x:(x/(1+.2*x*x))@mix+.04)]:
 cr={'ref_rosette':a,'raw_rosette':fn(a)}; inv=b._calibrator(cr); print('calibration',name,'RMS',rms(inv(fn(test)).reshape(-1,4)-test))
