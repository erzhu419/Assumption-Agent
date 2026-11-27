import importlib.util, pickle, sys
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=d['train']; c,_=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
for loss,f in [('linear',1),('soft_l1',1.5),('cauchy',2.5),('cauchy',4)]:
  tau,theta=m._fit(c,dt,t,y,loss,f); e=m._design(c,dt,t,tau)@theta-y
  print(loss,f,'tau',tau,'theta',theta,'fit med abs curve/rate',np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])))
  print('abs 50/90/99 curve,rate',np.quantile(abs(e[:,:-1]),[.5,.9,.99]),np.quantile(abs(e[:,-1]),[.5,.9,.99]))
  if loss=='cauchy' and f==2.5: print('MAD each time/rate',np.median(abs(e-np.median(e,axis=0)),axis=0))
  for l in np.unique(tr['lab_id']):
    ix=tr['lab_id']==l; a,b=m._fit(tuple(x[~ix] for x in c),dt,t,y[~ix],loss,f); r=abs(m._design(tuple(x[ix] for x in c),dt,t,a)@b-y[ix]); print('lab',l,'taus',a,'curve & rate median',np.median(r[:,:-1]),np.median(r[:,-1]),'short & long 90%', np.quantile(r[:,0],.9),np.quantile(r[:,-2],.9))
