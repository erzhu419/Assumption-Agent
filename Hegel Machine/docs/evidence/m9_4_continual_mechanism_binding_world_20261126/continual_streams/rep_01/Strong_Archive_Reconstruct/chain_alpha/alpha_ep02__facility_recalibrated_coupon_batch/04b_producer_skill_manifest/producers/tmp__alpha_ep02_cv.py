import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_ep02_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_ep02.pkl','rb'))[0]['dataset']
a,b,y,complete=m._prepare(d)
tr=d['train']; labs=tr['lab_id']; times=d['eval_times']; rates=y[:,-1]
for lab in [-1,*np.unique(labs)]:
 keep=labs!=lab
 tau,w=m._fit(a,y,complete & keep)
 pred=a.matrix(np.log(tau))@w
 for mask in [labs==lab if lab>=0 else keep]:
  r=(rates-pred[:,-1])[mask]; e=(y[:,:-1]-pred[:,:-1])[mask]
  print('lab',lab,'tau',np.round(tau,5),'w',np.round(w,5),'rate bias/MAD',np.round([np.median(r),np.median(abs(r))],5),'curve MAE/median per horizon',round(np.mean(abs(e)),6),np.round(np.median(abs(e),axis=0),5))
  for label,g in [('complete',mask&complete),('imputed',mask&~complete)]:
   if not np.any(g): continue
   res=y[g,:-1]-pred[g,:-1]; delta=rates[g]-pred[g,-1]
   print(label,np.sum(g),'median absolute late residual for rate updates 0,.25,.5,1',np.round([np.median(abs(res[:,-1]-k*times[-1]*delta)) for k in [0,.25,.5,1]],5),'rate abs median',round(np.median(abs(delta)),5))
