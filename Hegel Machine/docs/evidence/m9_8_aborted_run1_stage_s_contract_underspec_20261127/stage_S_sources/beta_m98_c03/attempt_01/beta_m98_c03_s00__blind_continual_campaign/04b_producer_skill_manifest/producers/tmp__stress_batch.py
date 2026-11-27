import importlib.util,pickle,numpy as np
from scipy.optimize import minimize
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['sensor_calibration_sweeps']; x=d['ref_rosette']; y=d['raw_rosette'];b=d['single_bank_sweeps']; print('sweep',x.shape,'quantiles',np.quantile(x,[0,.1,.5,.9,1],axis=0)); print('sweep coef',np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]);print('bank sweep shapes',{k:np.shape(v) for k,v in b.items()}); print('bank affine fits',[np.polyfit(b['ref_signal'][b['bank_id']==j],b['raw_readout'][b['bank_id']==j],1) for j in range(4)])
rng=np.random.default_rng(2026); mx=np.zeros(6); active=[0,0]
for i in range(400):
 n=int(rng.integers(2,33)); R=rng.normal(size=(n,n)); h=R.T@R/n+.04*np.eye(n);q=rng.normal(size=n)*3;u=np.broadcast_to(rng.uniform(.05,1.,size=n) if i%2 else .42,(n,)).copy()
 if i%11==0:u[0]=0
 frac=[0,1,1e-7,1-1e-7,.15,.5,.85][i%7];budget=u.sum()*frac
 w,nu=m._box_qp(h,q,u,budget);g=h@w+q+nu;l=np.where(w<=1e-10,np.maximum(g,0),0);v=np.where(w>=u-1e-10,np.maximum(-g,0),0);r=g-l+v
 err=0.
 if i%8==0:
  f=lambda z:.5*z@h@z+q@z
  sol=minimize(f,u*frac,jac=lambda z:h@z+q,bounds=list(zip(np.zeros(n),u)),constraints=[{'type':'eq','fun':lambda z:z.sum()-budget,'jac':lambda z:np.ones(n)}],method='SLSQP',options={'ftol':1e-11,'maxiter':300});err=abs(f(w)-f(sol.x))
 mx=np.maximum(mx,[abs(w.sum()-budget),max(0,-w.min(),(w-u).max()),abs(r).max(),abs(l@w).max(),abs(v@(u-w)).max(),err]);active[0]+=np.count_nonzero(l);active[1]+=np.count_nonzero(v)
print('QP worst [sum, box, stationarity, lower complementarity, upper complementarity, independent objective delta]',mx,'active counts',active);assert max(mx[:5])<1e-9 and mx[5]<1e-6
c=m._calibrator(d);t=m._Transport(s); p=s['reference_tracer_pulses'];wall=m._wall_fit(t,p,c)
for label,q in [('pulse',s),('blind',dict(s,reference_tracer_pulses=None))]:
 out=m.solve_batch(q); v=out['kkt_certificate']; used=wall if label=='pulse' else t.nominal; field=out['terminal_field'].ravel(order='F');from scipy.sparse import diags
 mat=t.matrix+diags(t.walls@used)+t.shifts[-1]*t.identity
 print(label,'PDE infinity residual',np.max(np.abs(mat@field-t.modes@out['weights'])),'duality gap',v['primal_objective']-v['dual_lower_bound'],'nominal vs tracer wall max relative delta',np.max(np.abs(t.nominal/wall-1)))
 assert np.max(np.abs(mat@field-t.modes@out['weights']))<1e-10 and abs(v['primal_objective']-v['dual_lower_bound'])<1e-10
