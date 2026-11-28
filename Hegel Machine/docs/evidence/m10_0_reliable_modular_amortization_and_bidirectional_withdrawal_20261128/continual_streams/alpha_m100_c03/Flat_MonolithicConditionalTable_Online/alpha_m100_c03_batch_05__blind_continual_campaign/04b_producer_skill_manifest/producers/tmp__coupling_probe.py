import pickle,json,numpy as np
from scipy.optimize import minimize_scalar
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; y=c['true_channel_loads']; r=c['bridge_raw']; keep=abs(r).max(1)<5; y,r=y[keep],r[keep]
pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
def basis(y,k,mode):
 s=np.sin(k*y); X=np.c_[np.ones(len(y)),y,s]; P=np.stack([y[:,i]*y[:,j] for i,j in pairs],axis=1)
 if mode==0:return X
 if mode==1:return np.c_[X,P]
 if mode==2:return np.c_[X,P,y**2,y**3]
 if mode==3:return np.c_[X,P,np.cos(k*y)-1,*[np.sin(k*y[:,i]*y[:,j]) for i,j in pairs]]
 if mode==4:return np.c_[X,P,*[np.sin(k*(y[:,i]+y[:,j]))-s[:,i]-s[:,j] for i,j in pairs],*[np.sin(k*(y[:,i]-y[:,j]))-s[:,i]+s[:,j] for i,j in pairs]]
 if mode==5:return np.c_[X,P,*[y[:,i]*s[:,j] for i,j in pairs],*[s[:,i]*y[:,j] for i,j in pairs]]
for mode in range(6):
 def fit(m):
  def res(k):
   A=basis(y[m],k,mode); b=np.linalg.lstsq(A,r[m],rcond=None)[0]; return np.linalg.norm(A@b-r[m]),b
  o=min([minimize_scalar(lambda k:res(k)[0],bounds=b,method='bounded') for b in [(1.8,2.45),(2.45,2.55),(2.55,3.3)]],key=lambda o:o.fun); return o.x,res(o.x)[1]
 k,b=fit(np.ones(len(y),bool)); e=basis(y,k,mode)@b-r
 ee=np.zeros_like(e)
 for i in range(5):
  m=np.arange(len(y))%5!=i; k0,b0=fit(m); ee[~m]=basis(y[~m],k0,mode)@b0-r[~m]
 print(mode,round(k,6),'fit',np.sqrt(np.mean(e**2)), 'cv',np.sqrt(np.mean(ee**2)), 'cvmax',abs(ee).max())
 if mode in (0,1): print(b.round(5))
 if mode==1: print('resid quantiles by ch',np.quantile(abs(e),[.5,.9,1],axis=0).round(5))
print('facility channel 3,6 means by campaign')
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print(row['completed_campaign_index'],row['completed_campaign_short_name'],np.mean(row['facility_telemetry_trace'],axis=0)[[3,6]].round(3))
