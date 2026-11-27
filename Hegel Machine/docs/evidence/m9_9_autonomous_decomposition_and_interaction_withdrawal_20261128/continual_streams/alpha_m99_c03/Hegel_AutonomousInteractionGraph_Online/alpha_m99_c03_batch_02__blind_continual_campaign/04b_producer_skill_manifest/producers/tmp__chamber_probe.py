import sys, pickle, json, importlib.util
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c03_batch_01_solver.py'); q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
d=calibrate_upstream(raw,pay); diag=calibrate_upstream(dict(raw,test=item['campaign_diagnostics']),pay)['test']
a,b=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; x=q._features(a,b); xd=q._features(a,diag)[len(a['lab_id']):]; xa=x[:len(a['lab_id'])]
def inp(block,x):
 return block['past_channel_raw'].sum(-1), block['hold_channel_raw'].sum(-1), np.column_stack((np.ones(len(x)),x))
def obs(block):
 return block['strain_curves_obs']-block['strain_0'][:,None],block['instant_rate_obs']
taus,theta=q._fit(inp(a,xa),dt,t,*obs(a)); print('intrinsic',taus,theta)
for block,features in [(a,xa),(diag,xd)]:
 P,H,X=inp(block,features); T,Y=obs(block); y=np.column_stack((T,Y)); r=-np.expm1(-t[:,None]/taus); k=np.vstack((np.column_stack((r,t)),np.r_[1/taus,1]));
 base=np.concatenate((P,H[:,None],X),axis=1); bh=np.linalg.lstsq(base,y,rcond=None)[0]; residual=y-base@bh
 print('free reg residual RMSE',np.sqrt(np.mean(residual**2,axis=0)).round(4))
 print('projected coeff rows=past6,hold,intercept,feat4; cols=relax2,linear'); print(np.linalg.lstsq(k,bh.T,rcond=None)[0].T.round(5)); print('outside temporal span',np.linalg.norm(bh-bh@k@np.linalg.pinv(k)) if False else np.sqrt(np.mean((bh-(bh@np.linalg.pinv(k.T))@k.T)**2)))
 after=np.cumsum(dt[::-1])[::-1]-dt; z=P@(-np.expm1(-dt[:,None]/taus)*np.exp(-after[:,None]/taus)); dr=X@theta[2:]; design=np.stack([np.outer(f,g) for f,g in [(H,k[:,0]*theta[0]),(H,k[:,1]*theta[1]),(z[:,0],k[:,0]*theta[0]),(z[:,1],k[:,1]*theta[1]),(dr,k[:,0]*taus[0]),(dr,k[:,1]*taus[1]),(dr,k[:,2])]],axis=2); A=design.reshape(-1,7); vv=y.ravel(); guess=np.linalg.lstsq(A,vv,rcond=None)[0]; fit=least_squares(lambda p:(A@p-vv),guess,loss='soft_l1',f_scale=.035); print('structured',fit.x.round(5),'rmse curve,rate',np.sqrt(np.mean(fit.fun.reshape(y.shape)[:,:-1]**2)),np.sqrt(np.mean(fit.fun.reshape(y.shape)[:,-1]**2)))
