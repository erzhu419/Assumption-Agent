import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=i['campaign_diagnostics']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; train,diag=m._prepare(dict(d,test=c)); y=np.column_stack([tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]); tau,theta=m._fit(train,dt,t,y)
yc=np.column_stack([c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]); pred=m._design(diag,dt,t,tau)@theta; r=yc-pred; past,h,f=diag; drift=f@theta[2:]; g=m._design(diag,dt,t,tau)[:,-1,:2]*tau; z=h[:,None]-g
rm=lambda a:np.sqrt(np.mean(a*a,axis=0))
print('residual/deployment ranges');print(np.round(np.c_[c['lab_id'],h,z,drift,r[:,[0,5,-2,-1]]],4)); print('rm by t',rm(r))
E=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); T=lambda p:np.r_[t**p,float(p==1)]; time={'E0':E[:,0],'E1':E[:,1],'t':T(1),'t2':T(2),'t3':T(3)}
vectors={'1':np.ones(len(h)),'r':drift,'h':h,'z0':z[:,0],'z1':z[:,1],'g0':g[:,0],'g1':g[:,1],'r2':drift**2,'h2':h*h,'hr':h*drift,'last':past[:,-1]}
print('top corrections relative to baseline:')
results=[]
for a,v in vectors.items():
 for b,w in time.items():
  A=v[:,None]*w; q=np.sum(A*r)/np.sum(A*A); e=r-A*q;results.append((np.sqrt(np.mean(e**2)),a,b,q,rm(e[:,-1])))
for row in sorted(results)[:16]: print(row)
print('per-coupon temporal decomposition: coefficient correction against baseline, RMS remaining')
for cols in [('E0','E1','t'),('E0','E1','t','t2'),('E0','E1','t','t2','t3')]:
 A=np.column_stack([time[k] for k in cols]); p=np.linalg.lstsq(A,r.T,rcond=None)[0].T;print(cols,np.round(rm((r-p@A.T).T),4));print(np.round(p,3));print('correlations',list(vectors),np.round(np.corrcoef(np.column_stack((p,*vectors.values())).T)[:len(cols),len(cols):],3))
