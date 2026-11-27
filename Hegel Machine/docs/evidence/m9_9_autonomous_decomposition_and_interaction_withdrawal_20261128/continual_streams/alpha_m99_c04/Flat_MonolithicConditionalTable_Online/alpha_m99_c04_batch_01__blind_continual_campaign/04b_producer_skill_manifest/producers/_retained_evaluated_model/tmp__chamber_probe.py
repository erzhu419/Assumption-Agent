import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_m99_c04_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr=d['train']; te=d['test']; dg=i['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; st=m._stress_map(d['calibration_load_cells']); xt,xe=m._features(tr,te); xd=m._features(tr,dg)[1]; tau,th=m._fit(tr,xt,st,dt,t); print('material',tau,th)
for g,x in [(tr,xt),(dg,xd)]:
 c,r=m._design(g,x,st,dt,t,tau); y=g['strain_curves_obs']-g['strain_0'][:,None]; v=g['instant_rate_obs']; e=y-c@th; f=v-r@th
 print('group',len(x),'curve/rate MAD',m._mad(e),m._mad(f),'RMSE',np.mean(e**2)**.5,np.mean(f**2)**.5,'curve residual RMS by time',np.round(np.mean(e**2,axis=0)**.5,5)); print('rate residual by lab',[(l,np.mean(f[g['lab_id']==l]),np.mean(f[g['lab_id']==l]**2)**.5) for l in np.unique(g['lab_id'])]);
 if g is tr: continue
 past=st[0]+g['past_channel_raw']@st[1:]; hold=st[0]+g['hold_channel_raw']@st[1:]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0]; z=past@(-np.expm1(-dt[:,None]/tau)*np.exp(-age[:,None]/tau)); A=np.column_stack([np.ones(len(x)),x,hold,past,z]); noms=['1','x0','x1','x2','x3','hold']+['p'+str(j) for j in range(len(dt))]+['z0','z1'];
 print('rate residual linear',dict(zip(noms,np.round(np.linalg.lstsq(A,f,rcond=None)[0],5))));
 def separate(ta):
  b=np.column_stack([-np.expm1(-t[:,None]/ta),t]); br=np.r_[1/ta,1.]; D=np.vstack([b,br*.4]); yy=np.column_stack([y,v*.4]); p=np.linalg.lstsq(D,yy.T,rcond=None)[0].T; return (p@D.T-yy),p
 print('baseline basis unconstrained rms',np.mean(separate(tau)[0]**2)**.5)
 for start in [tau,[.3,3]]:
  opt=least_squares(lambda a: separate(np.exp(a))[0].ravel(),np.log(start),bounds=(np.log(.051),np.log(24.99)),loss='soft_l1',f_scale=.008); ta=np.exp(opt.x); E,P=separate(ta); print('free time constants',ta,'rms',np.mean(E**2)**.5,'MAD',m._mad(E)); print('component regressions',np.round(np.linalg.lstsq(A,P,rcond=None)[0],5).T)
 print('feature mismatch from diagnostic vs train+test ref',np.nanstd(xd,axis=0),np.nanstd(xt,axis=0));
