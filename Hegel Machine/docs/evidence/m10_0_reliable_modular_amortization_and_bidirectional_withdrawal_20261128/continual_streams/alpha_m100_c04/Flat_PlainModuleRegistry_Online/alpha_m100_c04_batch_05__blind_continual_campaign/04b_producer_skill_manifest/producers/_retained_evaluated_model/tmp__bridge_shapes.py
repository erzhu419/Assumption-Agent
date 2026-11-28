import pickle, numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; s=c['single_channel_sweeps']; x=c['true_channel_loads']; y=c['bridge_raw']; g=np.max(abs(y),axis=1)<5
families={
 'tanh+linear': (lambda t,k:[t,np.tanh(k*t)],1),
 'tanh+cubic': (lambda t,k:[t,t**3,np.tanh(k*t)],1),
 'tanh+sin': (lambda t,k:[t,np.tanh(k[0]*t),np.sin(k[1]*t)],2),
 'sin+linear': (lambda t,k:[t,np.sin(k*t)],1),
 'sin+cubic': (lambda t,k:[t,t**3,np.sin(k*t)],1),
 'rational+cubic': (lambda t,k:[t,t**3,t/(1+(k*t)**2)],1),
 'atan+cubic': (lambda t,k:[t,t**3,np.arctan(k*t)],1),
 'erf+cubic': (lambda t,k:[t,t**3,__import__('scipy').special.erf(k*t)],1),
 'odd 7': (lambda t,k:[t,t**3,t**5,t**7],0),
}
for channel in range(4):
 m=(s['lab_id']==channel)&(abs(s['cmd_stress'])<5); t=s['true_stress'][m]; z=s['cmd_stress'][m]; print('channel',channel)
 for name,(basis,nk) in families.items():
  def fit(m):
   def profile(k):
    a=np.column_stack([np.ones(len(t)),*basis(t,k)]); b=np.linalg.lstsq(a[m],z[m],rcond=None)[0]; return a@b-z,b
   if nk:
    p=min((least_squares(lambda k:profile(k)[0][m],np.full(nk,a),bounds=(.08,8.),max_nfev=120) for a in [0.7,1.5,2.6,4.1]),key=lambda p:p.cost).x
   else: p=[]
   err,b=profile(p); return err,b,p
  e,b,k=fit(np.ones(len(t),bool)); loo=np.array([fit(np.arange(len(t))!=j)[0][j] for j in range(len(t))]); print(name, 'rmse %.3g LOO max %.3g'%(np.mean(e**2)**.5,max(abs(loo))),np.round(b,6),np.round(k,5))
