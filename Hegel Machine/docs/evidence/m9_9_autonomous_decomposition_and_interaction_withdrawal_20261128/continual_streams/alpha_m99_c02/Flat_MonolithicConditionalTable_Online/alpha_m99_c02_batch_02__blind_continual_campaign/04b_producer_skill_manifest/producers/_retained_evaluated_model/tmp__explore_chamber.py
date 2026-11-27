import pickle,numpy as np,importlib.util
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/alpha_m99_c02_batch_01_solver.py'
s=importlib.util.spec_from_file_location('prior',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; b=m._bridge(d['calibration_load_cells']); xtr,xte=m._features(tr,te); taus,theta=m._material(tr,b,xtr,dt,t)
cal=d['calibration_load_cells'];print('bridge',b,'total residual rms',np.std(cal['true_channel_loads'].sum(1)-(cal['bridge_raw']@b[1:]+b[0])))
def diag_features(ref):
 x=g['coupon_features_raw'].copy(); labs=g['lab_id']; src={k:np.concatenate([z[k] for z in ref]) for k in ['lab_id','ref_coupons']}
 for l in np.unique(labs):
  panel=src['ref_coupons'][src['lab_id']==l]; x[labs==l]=(x[labs==l]-np.median(panel.mean(0)))/np.median(panel.std(0))
 _,cell,cnt=np.unique(np.c_[labs,g['stratum']],axis=0,return_inverse=True,return_counts=True)
 for col in x.T:
  good=np.isfinite(col); col[~good]=np.average(col[good],weights=1/cnt[cell[good]])
 return x
x=diag_features([g]); print('diag missing',np.isnan(g['coupon_features_raw']).sum(0),'labs',np.unique(g['lab_id'],return_counts=True));past,hold=m._loads(g,b); gap=m._basis(taus,past,hold,x,dt,t)[1][:,:2]*taus; z=hold[:,None]-gap; drift=theta[2]+x@theta[3:]; y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; timebasis=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1]])
def report(k,e):
 print(k,'curve/rate mae',np.mean(abs(e[:,:-1])).round(5),np.mean(abs(e[:,-1])).round(5),'median',np.median(abs(e),axis=0).round(4),'lab worst',max(np.mean(abs(e[g['lab_id']==l])) for l in np.unique(g['lab_id'])).round(5))
report('base',np.c_[m._basis(taus,past,hold,x,dt,t)[0]@theta,m._basis(taus,past,hold,x,dt,t)[1]@theta]-y)
report('prior',np.column_stack(m._chamber(taus,theta,hold,x,t))-y)
base=np.c_[hold[:,None]*theta[:2],np.zeros(len(hold))]@timebasis.T
D=np.c_[np.ones(len(hold)),hold,z,drift,x,past];
for cols in ([4],[2,3,4],list(range(5)),list(range(9)),list(range(15))):
 A=(D[:,None,cols,None]*timebasis[None,:,None,:]).reshape(y.size,-1); target=(y-base).ravel(); q0=np.linalg.lstsq(A,target,rcond=None)[0]; q=least_squares(lambda v:A@v-target,q0,loss='soft_l1',f_scale=.004).x; report(str(cols), (A@q-target).reshape(y.shape));print(q.reshape(-1,3).round(4))
print('residual against hold zero history branches',np.quantile(base-y,[0,.25,.5,.75,1],axis=0).round(3));print('corr',np.corrcoef(np.c_[hold,z,drift,y[:,0]-base[:,0],y[:,-1]-base[:,-1]].T).round(3))
