import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m100_c04_s00_solver as s
np.set_printoptions(precision=5,suppress=True)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; p=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; cal=s._calibration(c); n=len(tr['lab_id']); x=s._features(tr,te); ti=s._inputs(tr,x[:n],cal); tau,theta=s._fit(ti,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau theta cal',tau,theta,cal)
r=c['bridge_raw']; loads=c['true_channel_loads']; A=np.c_[r,np.ones(len(r))]; Q=np.c_[A,*[r[:,k]*r[:,j] for k in range(4) for j in range(k,4)]]
for mat in [A,Q]:
 err=(mat@np.linalg.lstsq(mat,loads,rcond=None)[0]-loads)/(1-np.sum(np.linalg.qr(mat,mode='reduced')[0]**2,axis=1))[:,None]; print('bridge LOO RMSE chan and sum',np.sqrt(np.mean(err**2,axis=0)),np.sqrt(np.mean(err.sum(1)**2)))
print('cmd sweeps by lab slopes', [np.polyfit(c['single_channel_sweeps']['cmd_stress'][c['single_channel_sweeps']['lab_id']==j],c['single_channel_sweeps']['true_stress'][c['single_channel_sweeps']['lab_id']==j],1) for j in np.unique(c['single_channel_sweeps']['lab_id'])]);
px=s._features(tr,p)[n:]; past,h,cov=s._inputs(p,px,cal); a,b=s._design(tau,(past,h,cov),dt,t); Y=np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]; Y0=np.c_[a@theta,b@theta]
def basis(ta): return np.r_[np.c_[-np.expm1(-t[:,None]/ta),t],[[1/ta[0],1/ta[1],1]]]
def temporal(ta):
 D=basis(ta); return Y.T-D@np.linalg.lstsq(D,Y.T,rcond=None)[0]
ta=least_squares(lambda z:temporal(z).ravel(),tau,bounds=(.05,25)).x; print('diagnostic temporal fit',ta,'rmse old/new',*[np.sqrt(np.mean(temporal(z)**2)) for z in [tau,ta]])
D=basis(tau); V=np.linalg.lstsq(D,(Y-Y0).T,rcond=None)[0].T; Yb=Y-(cov@theta[2:])[:,None]*np.r_[t,1]; gains=np.linalg.lstsq(D,Yb.T,rcond=None)[0].T; print('residual SVD',np.linalg.svd(Y-Y0,compute_uv=False)[:8]); print('baseline rmse',np.sqrt(np.mean((Y-Y0)**2,axis=0))); print('diagnostic source ranges',np.min(past,axis=0),np.max(past,axis=0)); print('test source ranges',*[np.round(getattr(np,k)(s._inputs(te,x[n:],cal)[0],axis=0),3) for k in ['min','max']]);
L=np.eye(4)[p['lab_id']][:,1:]; S=np.eye(3)[p['stratum']][:,1:]; one=np.ones(len(past)); a0=b[:,:2]*tau; gap=a0; delta=np.diff(np.c_[past,h],axis=1); raw=p['past_channel_raw']; hold=p['hold_channel_raw']; cols={'past+hold':np.c_[past,h], 'gaps':gap,'cov':px,'lab':L,'stratum':S,'clipped':np.maximum(past,0),'squared':past**2,'signed-square':past*abs(past),'positive-delta':np.maximum(delta,0),'negative-delta':np.minimum(delta,0),'raw hold':hold,'past extrema':np.c_[past.min(1),past.max(1),past.std(1)],'hold*gap':h[:,None]*gap}
for name,z in cols.items():
 print('corr',name,np.round(np.corrcoef(np.c_[z,V].T)[:-3,-3:],2).T)
def cv(Z,W):
 pred=np.zeros_like(Y)
 for j in range(len(Y)):
  mask=np.arange(len(Y))!=j; pred[j]=Z[j]@np.linalg.lstsq(Z[mask],W[mask],rcond=None)[0]@D.T
 er=pred-(Y-Y0); return np.round([np.sqrt(np.mean(er[:,:-1]**2)),np.sqrt(np.mean(er[:,-1]**2)),np.max(np.sqrt(np.mean(er[:,:-1]**2,axis=1)))],5)
for keys in [[],['past+hold'],['gaps'],['cov'],['lab'],['past+hold','cov'],['past+hold','lab'],['past+hold','squared'],['past+hold','clipped'],['past+hold','positive-delta'],['past+hold','hold*gap'],['past+hold','past extrema'],['past+hold','raw hold']]:
 Z=np.column_stack([one]+[cols[k] for k in keys]); print('LOO',keys,cv(Z,V));
