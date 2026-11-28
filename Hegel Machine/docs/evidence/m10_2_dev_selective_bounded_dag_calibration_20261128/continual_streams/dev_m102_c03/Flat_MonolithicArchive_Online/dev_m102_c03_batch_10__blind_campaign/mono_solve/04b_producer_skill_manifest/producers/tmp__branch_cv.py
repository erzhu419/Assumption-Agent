import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; u=s['panel_c']['u']; t=s['eval_times']; n=len(tr['x']); z=m._calibrate(s['panel_a'],np.vstack([tr['x'],s['test']['x']]))
funcs={'soft':lambda x,p:x/(1+abs(x*p)),'tanh':lambda x,p:np.tanh(x*p),'sin':lambda x,p:np.sin(x*p),'cos':lambda x,p:np.cos(x*p),'gauss':lambda x,p:np.exp(-(x*p)**2),'soft2':lambda x,p:x/(1+(x*p)**2),'soft3':lambda x,p:x/np.sqrt(1+(x*p)**2)}
features={}
for key,names in [('v1',list(funcs)[:3]+['soft2','soft3']),('v2',['cos','gauss'])]:
 v=s['panel_c'][key]
 for name in names:
  for joint in (False,True):
   if not joint and key=='v1': continue
   f=funcs[name]
   def des(x,p):return np.c_[m._aff(x),f(x,p)] if joint else m._aff(f(x,p))
   def lin(p):return np.linalg.lstsq(des(u,p),v,rcond=1e-10)[0]
   best=None
   for start in (.6,1.4,2.4):
    opt=least_squares(lambda p:(des(u,p)@lin(p)-v).ravel(),np.full(4,start),bounds=(.1,4),gtol=1e-12)
    if best is None or opt.cost<best.cost:best=opt
   print(key,name,joint,best.x,np.sqrt(2*best.cost/v.size)); features[key,name,joint]=des(z,best.x)@lin(best.x)
for f1 in ('soft','tanh','sin','soft2','soft3'):
 for f2,j in (('cos',False),('gauss',True)):
  c=features['v1',f1,True]; d=features['v2',f2,j]; F=m._design(z,c,d)
  print(f1,f2,end=': ')
  for ix,jx in [(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.r_[0:26,46:72],np.arange(26,46))]:
   Y=tr['traj'][ix]; rates=least_squares(lambda p:(m._time(t,p)@np.linalg.lstsq(m._time(t,p),Y.T,rcond=None)[0]-Y.T).ravel(),np.log([.4,1.8]),bounds=(np.log([.12,1.]),np.log([.9,4.]))).x; Q=m._time(t,rates); c0=np.linalg.lstsq(F[ix],np.c_[tr['y'][ix],Y],rcond=1e-9)[0]; pred=F[jx]@c0; P=np.linalg.qr(Q)[0]; e=abs(pred[:,1:]@P@P.T-tr['traj'][jx]); print(np.round([np.mean(abs(pred[:,0]-tr['y'][jx])),e.mean(),max(e.mean(axis=0)),*np.exp(rates)],5),end=' ')
  print()
