import numpy as np, sys, pickle
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m99_c02_s00_solver as s
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=s._bridge(d['calibration_load_cells']); x,xt=s._features(tr,te); _,xd=s._features(tr,di); taus,w=s._material(tr,b,x,dt,t); p,h=s._loads(di,b); C,R=s._basis(taus,p,h,xd,dt,t); g=R[:,:2]*taus; y=di['strain_curves_obs']-di['strain_0'][:,None]; r=di['instant_rate_obs']; Y=np.c_[y,r]; B=np.vstack((np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1])); n=len(h); lab=di['lab_id']; drift=R[:,2:]@w[2:]; one=np.ones(n); g0,g1=g.T; z0,z1=(h[:,None]-g).T
base=np.c_[h*w[0],h*w[1]]
amp=Y@np.linalg.pinv(B[:,:2].T)
print('ratio corrections: (amp/theta - hold)/drift')
print(np.round(np.c_[h,g,z0,z1,drift,(amp/w[:2]-h[:,None])/drift[:,None]],3))
for fn,F in {'r':drift[:,None], 'h*r':(h*drift)[:,None], 'r+h*r':np.c_[drift,h*drift], 'r+g*r':np.c_[drift,g*drift[:,None]], 'r+h*r+g*r':np.c_[drift,h*drift,g*drift[:,None]], 'r+h*r+r²':np.c_[drift,h*drift,drift**2], 'r+h*r+g*r+r²':np.c_[drift,h*drift,g*drift[:,None],drift**2], 'h*r+r²+h*r²':np.c_[h*drift,drift**2,h*drift**2], 'r+g*r+z0*z1*r':np.c_[drift,g*drift[:,None],z0*z1*drift], 'r+g*r+h*r/(1+r²)':np.c_[drift,g*drift[:,None],h*drift/(1+drift**2)]}.items():
 for anchor in [True,False]:
  G=F if anchor else np.c_[h,F]; b0=base@B[:,:2].T if anchor else np.zeros_like(Y)
  N=(G[:,None,None,:]*B[None,:,:2,None]).reshape(n,len(t)+1,-1); E=np.empty_like(Y); scores=[]
  for l in np.unique(lab):
   m=lab!=l; fit=np.linalg.lstsq(N[m].reshape(-1,N.shape[-1]),(Y-b0)[m].ravel(),rcond=None)[0]; E[~m]=(N[~m]@fit+b0[~m]-Y[~m]); scores.append(np.sqrt(np.mean(E[~m]**2,axis=0)))
  fit=np.linalg.lstsq(N.reshape(-1,N.shape[-1]),(Y-b0).ravel(),rcond=None)[0]; print(fn,anchor,'fit/LOLO curve/rate/worst-time',*np.round([np.std(N@fit+b0-Y),np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),np.max(scores)],5),'coeff',np.round(fit.reshape(2,-1)/w[:2,None],4))
for label,a in [('train',tr),('test',te),('diag',di)]:
 p2,h2=s._loads(a,b); print(label,'hold',np.quantile(h2,[0,.5,1]));print('reference summary',[(int(l),np.round(np.median(np.mean(a['ref_coupons'][a['lab_id']==l],axis=0)),4),np.round(np.median(np.std(a['ref_coupons'][a['lab_id']==l],axis=0)),4)) for l in np.unique(lab)])
