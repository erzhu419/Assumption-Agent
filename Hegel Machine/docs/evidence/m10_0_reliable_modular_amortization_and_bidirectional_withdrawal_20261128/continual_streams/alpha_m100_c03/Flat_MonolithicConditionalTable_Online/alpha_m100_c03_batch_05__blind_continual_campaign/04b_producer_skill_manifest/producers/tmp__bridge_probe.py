import pickle,numpy as np
from scipy.optimize import minimize_scalar
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; y=c['true_channel_loads']; r=c['bridge_raw']; keep=abs(r).max(1)<5; y,r=y[keep],r[keep]; pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
for i in range(4):
 m=np.count_nonzero(y,axis=1)==1;m &= y[:,i]!=0
 print('axis',i,np.c_[y[m,i],r[m]].round(5))

def basis(y,k,mode):
 s=np.sin(k*y); p=np.array([y[:,i]*y[:,j] for i,j in pairs]).T
 a=np.c_[np.ones(len(y)),y,s]
 if mode==0:return np.c_[a,np.cos(k*y)-1]
 if mode==1:return np.c_[a,np.cos(k*y)-1,p]
 if mode==2:return np.c_[a,np.cos(k*y)-1,p,y**2]
 if mode==3:return np.c_[a,np.abs(y),p]
 if mode==4:return np.c_[a,np.logaddexp(y,-y)-np.log(2),p]
 if mode==5:return np.c_[a,np.tanh(y),y*np.tanh(y),p]
 if mode==6:return np.c_[a,np.cos(k*y)-1,p,*[np.cos(k*(y[:,i]-y[:,j]))-np.cos(k*y[:,i])-np.cos(k*y[:,j])+1 for i,j in pairs]]
 if mode==7:return np.c_[a,np.cos(k*y)-1,p,*[np.cos(k*(y[:,i]+y[:,j]))-np.cos(k*y[:,i])-np.cos(k*y[:,j])+1 for i,j in pairs]]
for mode in range(8):
 def fit(m):
  def res(k):
   A=basis(y[m],k,mode); b=np.linalg.lstsq(A,r[m],rcond=None)[0]; return np.linalg.norm(A@b-r[m]),b
  o=minimize_scalar(lambda k:res(k)[0],bounds=(2.15,2.85),method='bounded');return o.x,res(o.x)[1]
 k,b=fit(np.ones(len(y),bool)); e=basis(y,k,mode)@b-r; ee=np.zeros_like(e)
 for i in range(5):
  m=np.arange(len(y))%5!=i; kk,bb=fit(m); ee[~m]=basis(y[~m],kk,mode)@bb-r[~m]
 print(mode,round(k,5),'fit',np.linalg.norm(e)/np.sqrt(e.size),'CV',np.linalg.norm(ee)/np.sqrt(ee.size),'max',abs(ee).max());
 if mode in (0,1,3,4,5): print(b.round(4))
