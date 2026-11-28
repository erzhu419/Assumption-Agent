import pickle,runpy,glob
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]);lin=a['lin'];one=a['one']
u,v=[s['panel_a'][k] for k in ('u','v')];o,h=a['_align'](s['panel_a']);print('align offset scale',o,h)
for j in range(u.shape[1]):
 x=u[:,j];e=v[:,j]-o[j]-x*h[j]; ix=np.argsort(x);print('alignment x,resid',j,np.round(np.c_[x,e][ix][::2],5).tolist());print('quantiles',np.quantile(abs(e),[.1,.5,.75,.9,1]))
u=s['panel_c']['u'];w=a['_panel_maps'](s['panel_c'],u);print('archived panel residual RMSE',np.sqrt(np.mean((w-np.c_[s['panel_c']['v1'],s['panel_c']['v2']])**2,0)));print('C range',u.min(0),u.max(0));print('z range',((np.r_[s['train']['x'],s['test']['x']]-o)/h).min(0),((np.r_[s['train']['x'],s['test']['x']]-o)/h).max(0))
for key in ('v1','v2'):
 Y=s['panel_c'][key];print(key)
 for k in range(1,5):
  F=one(np.concatenate([u**i for i in range(1,k+1)],axis=1)); print('degree',k,'rmse',np.round(np.sqrt(np.mean((F@lin(F,Y)-Y)**2,0)),7))
 models={'soft':(lambda x,q:one(x/(1+q[0]*abs(x))+q[1]*x),[.85,0],[.001,-2],[20,2]),'rational2':(lambda x,q:np.c_[one(x),x*x,x*x/(1+q[0]*x*x)],[.25],[.0001],[20]),'rationalodd':(lambda x,q:np.c_[one(x),x/(1+q[0]*x*x)],[.5],[.0001],[20]),'tanh':(lambda x,q:np.c_[one(x),np.tanh(q[0]*x)],[.8],[.001],[20]),'trig':(lambda x,q:np.c_[one(x),np.sin(q[0]*x),np.cos(q[0]*x)],[1.],[.001],[10]),'abs':(lambda x,q:np.c_[one(x),x*x,x*x/(1+q[0]*abs(x))],[.5],[.0001],[20]),'powers':(lambda x,q:np.c_[one(x),x*x,x*x/(1+q[0]*abs(x)**q[1])],[1.,2.],[.001,.1],[30.,8.])}
 for name,(basis,q0,lo,hi) in models.items():
  def resid(q):
   F=basis(u,q);return (F@lin(F,Y)-Y).ravel()
  q=least_squares(resid,q0,bounds=(lo,hi),gtol=1e-13).x;print(name,q,np.sqrt(np.mean(resid(q)**2)))
