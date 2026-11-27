import pickle, runpy
import numpy as np
s=runpy.run_path('/holdout_task/batch_solver.py')
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; cal=d['calibration_load_cells']; x=cal['true_channel_loads']; y=cal['bridge_raw']; keep=np.max(abs(y-np.median(y,axis=0)),axis=1)<8; x,y=x[keep],y[keep]
pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
for name in ('affine','adjacent+cubic','full quadratic+self cubic','adjacent+cubic+rational perturbation'):
 errors=[]
 for fold in range(4):
  test=np.arange(len(x))%4==fold
  for j in range(4):
   cols=[np.ones(len(x)),*x.T]; q=x[:,j]; r=x[:,(j+1)%4]
   if name=='full quadratic+self cubic': cols += [*(x*x).T,*[x[:,a]*x[:,b] for a,b in pairs],q**3]
   elif name!='affine': cols += [q*r,q**3]
   if name.endswith('perturbation'): cols += [q**5, q*r*(q-r)**2]
   A=np.column_stack(cols); c=np.linalg.lstsq(A[~test],y[~test,j],rcond=None)[0]; errors.extend(abs(A[test]@c-y[test,j]))
 print(name,'held-out raw p90/max',np.quantile(errors,.9),max(errors))
out=s['solve_batch'](d); u=out['estimated_taus']; w=out['hereditary_weights']; _,coupon=s['_prepare'](d)
zero,rate=s['_chamber_predict'](coupon,d['past_dt'],np.array([0.,1e-6]),u,w); h=1e-6
print('zero strain increment',np.max(abs(zero[:,0])),'initial derivative max discrepancy',np.max(abs((zero[:,1]-zero[:,0])/h-out['initial_rate_pred'])))
assert 0.05<u[0]<u[1]<25 and all(np.isfinite(v).all() for v in out.values())
assert np.max(abs(zero[:,0]))<1e-12 and np.max(abs((zero[:,1]-zero[:,0])/h-out['initial_rate_pred']))<1e-4
