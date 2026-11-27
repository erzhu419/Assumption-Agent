import numpy as np, sys, pickle
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m99_c02_s00_solver as s
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=s._bridge(d['calibration_load_cells']); x,xt=s._features(tr,te); _,xd=s._features(tr,di); taus,w=s._material(tr,b,x,dt,t); p,h=s._loads(di,b); C,R=s._basis(taus,p,h,xd,dt,t); g=R[:,:2]*taus; y=di['strain_curves_obs']-di['strain_0'][:,None]; r=di['instant_rate_obs']; Y=np.c_[y,r]; B=np.vstack((np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1])); n=len(h); lab=di['lab_id']; drift=R[:,2:]@w[2:]; one=np.ones(n); g0,g1=g.T; z0,z1=(h[:,None]-g).T
print('lab counts',np.unique(lab,return_counts=True)); print('tau/w',taus,w); print('ranges',np.quantile(np.c_[h,g,z0,z1,drift],[0,.5,1],axis=0));
for k in [2,3]:
 A=Y@np.linalg.pinv(B[:,:k].T); E=Y-A@B[:,:k].T; print('free amplitudes',k,'residual RMS/max',np.std(E),np.abs(E).max(),'amp and drift ranges',np.quantile(A,[0,.5,1],axis=0));
print('h,g0,g1,drift, free2 normalized amplitudes'); print(np.round(np.c_[h,g,drift,(Y@np.linalg.pinv(B[:,:2].T))/w[:2]],4))
models={'gap':g,'gap+1':np.c_[g,one],'gap+h':np.c_[g,h],'gap+h+1':np.c_[g,h,one],'gap+g0*g1':np.c_[g,g0*g1],'gap+z0*z1':np.c_[g,z0*z1],'gap+h+drift':np.c_[g,h,drift],'gap+h+quad':np.c_[g,h,g0*g0,g0*g1,g1*g1,g0*h,g1*h,h*h],'linear histories':np.c_[h,p],'gap+h+X':np.c_[g,h,xd],'gap+1+X':np.c_[g,one,xd]}
for label,F in models.items():
 for k in [2,3]:
  N=(F[:,None,None,:]*B[None,:,:k,None]).reshape(n,len(t)+1,-1); E=np.empty_like(Y)
  for l in np.unique(lab):
   m=lab!=l; fit=np.linalg.lstsq(N[m].reshape(-1,N.shape[-1]),Y[m].ravel(),rcond=None)[0]; E[~m]=N[~m]@fit-Y[~m]
  fit=np.linalg.lstsq(N.reshape(-1,N.shape[-1]),Y.ravel(),rcond=None)[0]; print(label,k,'fit/LOLO curve/rate/worst',*np.round([np.std(N@fit-Y),np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),np.abs(E).max()],5), 'coef',np.round(fit,4) if k==2 and F.shape[1]<=4 else '')
