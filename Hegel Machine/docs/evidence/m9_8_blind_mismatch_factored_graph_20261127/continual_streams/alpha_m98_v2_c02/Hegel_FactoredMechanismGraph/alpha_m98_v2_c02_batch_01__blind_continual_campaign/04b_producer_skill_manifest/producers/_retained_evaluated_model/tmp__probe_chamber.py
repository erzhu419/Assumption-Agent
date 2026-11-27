import pickle,numpy as np,glob,runpy
from scipy.optimize import least_squares
m=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); q=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=q['dataset']; D=q['campaign_diagnostics']; tr,te=d['train'],d['test']; S=m['_stress_calibration'](d['calibration_load_cells']); x,xt=m['_standardize'](tr,te); dt,t=d['past_dt'],d['eval_times']; tau,beta=m['_fit'](m['_design'](tr,x,S,dt,t),tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); h=S(D['hold_channel_raw']); p=S(D['past_channel_raw']); n=len(h); one=np.ones(n); y=np.c_[D['strain_curves_obs']-D['strain_0'][:,None],D['instant_rate_obs']]; B=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); A=np.linalg.lstsq(B,y.T,rcond=None)[0].T; r=(A-h[:,None]*beta[:2]); print('material',tau,beta); print('offset relation',np.linalg.lstsq(np.c_[one,h,r[:,0]],r[:,1],rcond=None)[0]); print('h, corrections, corrections/tau',np.round(np.c_[h,r,r/tau],4)); w=-np.expm1(-dt[:,None]/tau)*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/tau); z=p@w; zr=p[:,::-1]@w
keys=('coupon_features_raw','ref_coupons','lab_id','stratum'); Xs={'trdiag':m['_standardize'](tr,D)[1],'diag':m['_standardize'](D,D)[1],'all':m['_standardize'](tr,{k:np.concatenate((te[k],D[k])) for k in keys})[1][-n:]}; print('diag missing',np.where(np.isnan(D['coupon_features_raw']))); print('raw drift variants')
for key,X in Xs.items():
 rd=beta[2]+X@beta[3:]; print(key,np.round(rd,3)); out=[]
 for j in range(2):
  for nam,g in [('none',one[:,None]),('h',np.c_[one,h]),('allp',np.c_[one,p]),('chron',np.c_[one,z]),('rev',np.c_[one,zr]),('first',np.c_[one,p[:,0]]),('mean',np.c_[one,p@dt/dt.sum()]),('first3',np.c_[one,p[:,:3]]),('h+p',np.c_[one,h,p])]:
   for trans,f in [('linear',rd),('abs',abs(rd)),('exp',np.exp(rd)),('softplus',np.logaddexp(0,rd)),('tanh',np.tanh(rd)),('relu',np.maximum(rd,0)),('quad',np.c_[rd,rd**2]),('none',np.empty((n,0)))]:
    G=np.column_stack((g,f)); co=np.linalg.lstsq(G,r[:,j],rcond=None)[0]; e=G@co-r[:,j]; lev=np.sum(np.linalg.svd(G,full_matrices=False)[0]**2,axis=1); out.append((np.sqrt(np.mean((e/np.maximum(1-lev,1e-9))**2)),j,nam,trans,np.round(co,4)))
 for j in range(2): print('best',j,sorted((a for a in out if a[1]==j),key=lambda a:a[0])[:5]);
for site in ('train','test'):
 T=d[site]['aux_chamber_telemetry']; print('telemetry',site,np.round(T[:4],4),np.round(np.corrcoef(np.c_[T,S(d[site]['hold_channel_raw']),S(d[site]['past_channel_raw'])].T)[:4,4:],2))
