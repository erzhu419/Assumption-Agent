import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];a=i['campaign_diagnostics'];tr,te=d['train'],d['test'];t=d['eval_times'];out=s.solve_batch(d);ts,w=out['estimated_taus'],out['hereditary_weights'];b=s._bridge(d['calibration_load_cells']);p,h=s._loads(a,b);_,x=s._features({k:v[:0] for k,v in a.items()},a);r=w[2]+x@w[3:];g=s._basis(ts,p,h,x,d['past_dt'],t)[1][:,:2]*ts;P=np.vstack((-np.expm1(-t[:,None]/ts),1/ts));base=(h[:,None]*w[:2])@P.T;Y=np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']];L=a['lab_id']; c=d['calibration_load_cells'];raw=c['bridge_raw'];load=c['true_channel_loads'].sum(1);lin=np.c_[np.ones(len(raw)),raw];quad=np.c_[lin,*[raw[:,j]*raw[:,k] for j in range(4) for k in range(j,4)]]
for name,M in [('affine',lin),('quadratic',quad)]:
 e=np.empty_like(load)
 for k in range(5):
  m=np.arange(len(load))%5==k;e[m]=M[m]@np.linalg.lstsq(M[~m],load[~m],rcond=None)[0]-load[m]
 print('bridge',name,'OOF RMS/max',np.sqrt(np.mean(e**2)),np.max(abs(e)))
for name in ['baseline','pairwise','three_way','rational','linear_drift']:
 E=np.zeros_like(Y)
 for lab in np.unique(L):
  m=L==lab
  if name=='rational':
   def predict(q):return base+((r/(1+q[2]*r*r))[:,None]*q[:2])@P.T
   q=least_squares(lambda q:(predict(q)-Y)[~m].ravel(),[.37,2.08,.01],bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,10])).x
   pred=predict(q)
  else:
   delta=g[:,0]-g[:,1]; total=g.sum(1); features={'baseline':r[:,None],'pairwise':np.c_[r,delta,total],'three_way':np.c_[r,delta,total,r*delta,r*total,r*g[:,0]*g[:,1]],'linear_drift':r[:,None]}[name]
   A=(P[None,:,:,None]*features[:,None,None,:]).reshape(len(r),len(t)+1,-1)
   if name=='linear_drift': A=np.concatenate((A,r[:,None,None]*np.r_[t,1][None,:,None]),axis=2)
   q=np.linalg.lstsq(A[~m].reshape(-1,A.shape[-1]),(Y-base)[~m].ravel(),rcond=None)[0];pred=base+A@q
  E[m]=(pred-Y)[m]
 within=np.array([np.sqrt(np.mean(E[L==lab]**2,axis=0)) for lab in np.unique(L)]); biases=np.array([np.mean(E[L==lab],axis=0) for lab in np.unique(L)])
 print('chamber',name,'OOF curve rate worst lab-horizon max within-lab bias',np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),within.max(),abs(biases).max())
