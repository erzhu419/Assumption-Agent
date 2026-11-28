import pickle
import numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; T=c['true_channel_loads']; R=c['bridge_raw']; s=c['single_channel_sweeps']; ok=np.max(abs(R),axis=1)<5; T,R=T[ok],R[ok]; X=[]; Y=[]; G=[]
for j in range(4):
 m=(s['lab_id']==j)&(abs(s['cmd_stress'])<5); u=s['true_stress'][m]; v=s['cmd_stress'][m]; A=lambda x:np.c_[np.ones(len(x)),x,np.sin(2.5*x)]; b=np.linalg.lstsq(A(u),v,rcond=None)[0]; X.extend(T[:,j^2]); Y.extend((R[:,j]-A(T[:,j])@b)/b[1]); G.extend(range(len(T)))
x,y,g=map(np.array,(X,Y,G)); ii=np.argsort(x); print('collapsed paired response',np.round(np.c_[x[ii],y[ii]][::8],6));
O={'linear':lambda x,k:x,'tanh':lambda x,k:np.tanh(k*x),'sine':lambda x,k:np.sin(k*x),'rational':lambda x,k:x/(1+k*x*x),'atan':lambda x,k:np.arctan(k*x)}
E={'tanh2':lambda x,k:np.tanh(k*x)**2,'cos':lambda x,k:1-np.cos(k*x),'exp':lambda x,k:1-np.exp(-k*x*x),'rational':lambda x,k:x*x/(1+k*x*x),'logcosh':lambda x,k:np.log(np.cosh(k*x)),'sqrt':lambda x,k:np.sqrt(1+k*x*x)-1,'relu':lambda x,k:abs(x),'tanhx':lambda x,k:x*np.tanh(k*x)}
results=[]
for on,o in O.items():
 for en,e in E.items():
  def a(x,p):return np.c_[x,o(x,p[0]),e(x,p[1])]
  def prof(p,m):return np.linalg.lstsq(a(x[m],p),y[m],rcond=None)[0]
  def fit(m):
   ps=[least_squares(lambda p:a(x[m],p)@prof(p,m)-y[m],p,bounds=([.03,.03],[9,9]),max_nfev=100) for p in ([1,1],[2.5,2],[4,4])]; p=min(ps,key=lambda f:f.cost).x;return p,prof(p,m)
  m=g%5!=0; p,b=fit(m); z=y[~m]-a(x[~m],p)@b; results.append((np.sqrt(np.mean(z*z)),max(abs(z)),on,en,p,b))
for q in sorted(results,key=lambda z:z[0])[:15]:print(q)
