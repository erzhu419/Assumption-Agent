import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5; axis=np.count_nonzero(X,axis=1)==1; sb=c['single_bank_sweeps']; u,v,b=[sb[k] for k in ('ref_signal','raw_readout','bank_id')]; base=np.array([np.median(Y[clean&axis&(X[:,j]==0)&(X[:,j^2]==0),j]) for j in range(4)]); print('offset',base)
m=abs(v)<5; t=u[m]; z=v[m]-base[b[m]]; bank=b[m];
F={'tanh':lambda x,k:np.tanh(k*x),'atan':lambda x,k:np.arctan(k*x),'sqrt':lambda x,k:k*x/np.sqrt(1+(k*x)**2),'erf':lambda x,k:__import__('scipy').special.erf(k*x)}
for name,f in F.items():
 for term in ('none','lin','cub','lincub','sin','rat','sat2'):
  def D(x,p):
   a=[f(x,np.exp(p[0]))]; k=np.exp(p[-1]);
   a+= {'none':[], 'lin':[x],'cub':[x**3],'lincub':[x,x**3],'sin':[np.sin(k*x)],'rat':[x/(1+(k*x)**2)],'sat2':[f(x,k)]}[term]; return np.column_stack(a)
  def coeff(p):return np.stack([np.linalg.lstsq(D(t[bank==j],p),z[bank==j],rcond=None)[0] for j in range(4)])
  def err(p):return np.sum(D(t,p)*coeff(p)[bank],axis=1)-z
  init=[0.,0.4] if term in ('sin','rat','sat2') else [0.]; p=least_squares(err,init,bounds=(-3,2),max_nfev=400,gtol=1e-13).x; print(name,term,np.sqrt(np.mean(err(p)**2)),np.exp(p),np.round(coeff(p)[0],6))
# Interpolate the independently measured diagonal shapes, then isolate cross terms on mixed rows.
pair=np.arange(4)^2; C=[]; T=[]
for j in range(4):
 use=m&(b==j); x=np.r_[u[use],0]; y=np.r_[v[use]-base[j],0]; order=np.argsort(x); f=CubicSpline(x[order],y[order]); r=Y[:,j]-base[j]-f(X[:,j]); C.append(r[clean]); T.append(X[clean,j^2]);
C=np.array(C).T; T=np.array(T).T; D0=np.abs(X[clean]); print('cross isolated quantiles',np.round(np.quantile(C,[0,.1,.5,.9,1],axis=0),5));
for j in range(4):
 order=np.argsort(T[:,j]); zidx=order[np.linspace(0,len(order)-1,15,dtype=int)]; print(j,np.round(np.c_[T[zidx,j],C[zidx,j],D0[zidx,j]],5).tolist())
for name,f in F.items():
 for term in ('even','quad','rat','shift','linquad','coseven'):
  def D(x,p):
   k=np.exp(p[0]); a=f(x,k); l=np.exp(p[-1]); return np.stack({'even':[a,a*a],'quad':[a,x*x],'rat':[a,x*x/(1+(l*x)**2)],'shift':[a,f(x+p[-1],k)-f(p[-1],k)],'linquad':[x,a,x*x/(1+(l*x)**2)],'coseven':[a,1-np.cos(l*x)]}[term],axis=-1)
  def coeff(p):return np.stack([np.linalg.lstsq(D(T[:,j],p),C[:,j],rcond=None)[0] for j in range(4)])
  def err(p):return (np.sum(D(T,p)*coeff(p)[None,:,:],axis=2)-C).ravel()
  p=least_squares(err,[0.,0.6] if term not in ('even','quad') else [0.],bounds=(-4,2),max_nfev=300).x; print('cross',name,term,np.sqrt(np.mean(err(p)**2)),p,np.round(coeff(p)[0],6))
