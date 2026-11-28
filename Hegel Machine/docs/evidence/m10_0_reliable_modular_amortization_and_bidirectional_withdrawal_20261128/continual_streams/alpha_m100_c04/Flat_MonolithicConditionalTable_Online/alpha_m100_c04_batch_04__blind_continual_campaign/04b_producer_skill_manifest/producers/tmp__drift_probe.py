import pickle, numpy as np, runpy
from scipy.optimize import least_squares
s=runpy.run_path('/tmp/hegel_continual_archive/alpha_m100_c04_batch_03_solver.py'); it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=it['dataset']; d=it['campaign_diagnostics']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']; x=s['_features'](tr,te); xd=s['_features'](tr,d)[len(tr['lab_id']):]; c=ds['calibration_load_cells']; r,l=c['bridge_raw'],c['true_channel_loads']; print('cal max sorted',np.sort(np.max(abs(r),axis=1))[-15:]); print('sweeps',[(k,np.shape(v)) for k,v in c['single_channel_sweeps'].items()]); b=s['_calibrate'](c); print('cal coeff',b.round(5)); print('cal residual quantiles',np.quantile(np.max(abs(np.einsum('nck,ck->nc',s['_bridge_design'](l),b)-r),axis=1),[.25,.5,.75,.9,.95,1])); inp=s['_inputs'](tr,x[:len(tr['lab_id'])],b); tau,th=s['_fit'](inp,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th); p,h,z=s['_inputs'](d,xd,b); dr=z@th[2:]; y=d['strain_curves_obs']-d['strain_0'][:,None]; rate=d['instant_rate_obs']; a,g=s['_design'](tau,(p,h,z),dt,t); ar,gr=s['_chamber_response'](tau,th,h,z,t); print('baseline/reset rmse',[(np.sqrt(np.mean((v-y)**2)),np.sqrt(np.mean((w-rate)**2))) for v,w in [(a@th,g@th),(ar,gr)]]); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; state=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau)); R=-np.expm1(-t[:,None]/tau); T=np.vstack((np.c_[R,t],np.r_[1/tau,1])); Y=np.c_[y,rate]; print('diagnostic low-rank residual',np.quantile(abs(Y-np.linalg.lstsq(T,Y.T,rcond=None)[0].T@T.T),[.5,.9,.99,1])); print('hold and past',h[:5],p[:5]);
from scipy.optimize import least_squares
C0=np.c_[h[:,None]*th[:2],np.zeros(len(h))]; E=Y-C0@T.T
print('T',T.round(4)); print('reference fit details by lab')
for j in np.unique(d['lab_id']):
 for name,k in [('train',tr),('test',te),('diag',d)]:
  panel=k['ref_coupons'][k['lab_id']==j]; print(j,name, np.nanmedian(np.nanmean(panel,axis=0)).round(4),np.nanmedian(np.nanstd(panel,axis=0)).round(4))

def solve(D,Y):
 D=D.reshape(Y.size,-1); yy=Y.ravel(); q=np.linalg.lstsq(D,yy,rcond=None)[0]; return least_squares(lambda q:D@q-yy,q,jac=lambda q:D.copy(),loss='soft_l1',f_scale=.006).x

def test(name,D):
 q=solve(D,E); yh=D@q; pred=np.zeros_like(E)
 for j in np.unique(d['lab_id']):
  m=d['lab_id']==j; pred[m]=D[m]@solve(D[~m],E[~m]);
 err=pred-E; print(name,'params',q.round(5),'in mae',abs(yh-E).mean(0)[[0,5,11,12]].round(5),'LOLO MAE',abs(err).mean(0)[[0,5,11,12]].round(5),'lab RMSE',[(int(j),np.sqrt(np.mean(err[d['lab_id']==j,:-1]**2)).round(5),np.sqrt(np.mean(err[d['lab_id']==j,-1]**2)).round(5)) for j in np.unique(d['lab_id'])]);
 return q
q=test('shared drift',dr[:,None,None]*T[None,:,:]); test('drift+intercept',np.concatenate([dr[:,None,None]*T,T[None,:,:]*np.ones((len(h),1,1))],axis=2)); test('drift x hold',np.concatenate([dr[:,None,None]*T,(dr*(h-3.3))[:,None,None]*T],axis=2)); test('drift quadratic',np.concatenate([dr[:,None,None]*T,(dr**2)[:,None,None]*T],axis=2)); test('drift + material adjustment',np.concatenate([dr[:,None,None]*T, h[:,None,None]*T[None,:,:2]],axis=2)); test('drift free horizon',dr[:,None,None]*np.eye(len(T))[None,:,:]); test('features', (z[:,None,:,None]*T[None,:,None,:]).reshape(len(h),len(T),-1)); test('drift by lab level',np.concatenate([dr[:,None,None]*T,(np.eye(4)[d['lab_id']][:,None,:,None]*T[None,:,None,:]).reshape(len(h),len(T),-1)],axis=2));
fit=T@q; latent=E@fit/(fit@fit); print('effective drift delta',np.c_[d['lab_id'],dr,latent,latent-dr].round(5)); print('worst residual by coupon',np.max(abs(E-latent[:,None]*fit),axis=1).round(5));
