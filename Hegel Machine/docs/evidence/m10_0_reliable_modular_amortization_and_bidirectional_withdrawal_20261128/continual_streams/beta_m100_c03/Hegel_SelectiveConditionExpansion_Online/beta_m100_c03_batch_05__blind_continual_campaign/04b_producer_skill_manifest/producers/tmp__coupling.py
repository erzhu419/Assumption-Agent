import numpy as np,pickle
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import chebvander
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];c=s['sensor_calibration_sweeps'];x=c['ref_rosette']; y=c['raw_rosette'];d=c['single_bank_sweeps'];u=d['ref_signal'];bank=d['bank_id'];r=d['raw_readout']; g=np.max(abs(y),1)<6; h=abs(r)<6
A=np.vstack((x,np.eye(4)[bank]*u[:,None]));Y=np.vstack((y,np.where(np.eye(4,dtype=bool)[bank],r[:,None],np.nan)));G=np.r_[g,h]; Y[~G]=np.nan
js=np.array([2,3,0,1]);i,j=np.triu_indices(4,1)
def design(k,p,q,typ):
 t=A[:,k]; v=A[:,js[k]]; P=chebvander(t/1.8,p);Q=chebvander(v/1.8,q)[:,1:]-chebvander(np.array([0.]),q)[:,1:];
 return np.c_[P,Q,*({'pair':[A[:,i]*A[:,j]],'mod':[A[:,i]*A[:,j],(A[:,i]*A[:,j]**2),(A[:,i]**2*A[:,j])],'all':[np.concatenate([chebvander(A[:,l]/1.8,3)[:,1:] for l in range(4) if l not in (k,js[k])],axis=1)],'none':[]}[typ])]
for p,q,typ in [(p,q,typ) for p in [5,7,9,11] for q in [3,5,7] for typ in ['none','pair','mod','all']]:
 err=[];loo=[]
 for k in range(4):
  Z=design(k,p,q,typ);m=np.isfinite(Y[:,k]);H=Z[m];Yk=Y[m,k]; b=np.linalg.lstsq(H,Yk,rcond=None)[0];e=Yk-H@b; lev=np.sum(np.linalg.qr(H,mode='reduced')[0]**2,1); err.append(np.sqrt(np.mean(e**2)));loo.append(np.sqrt(np.mean((e[:g.sum()]/np.maximum(1-lev[:g.sum()],1e-10))**2)))
 print(max(loo),p,q,typ,np.round(err,7),np.round(loo,7))
for name,f,a in [('tanh+cubic_rat',lambda t,p:np.c_[np.ones(len(t)),np.tanh(p[0]*t),t**3/(1+p[1]*t*t)],[1,.4]),('tanh+linear_rat',lambda t,p:np.c_[np.ones(len(t)),np.tanh(p[0]*t),t/(1+p[1]*t*t)],[1,.4]),('tanh_over+linear',lambda t,p:np.c_[np.ones(len(t)),np.tanh(p[0]*t)/(1+p[1]*t*t),t],[1,.4]),('tanh+tanh',lambda t,p:np.c_[np.ones(len(t)),np.tanh(p[0]*t),np.tanh(p[1]*t)],[1,.4]),('tanh+even_rat',lambda t,p:np.c_[np.ones(len(t)),np.tanh(p[0]*t),t*t/(1+p[1]*t*t)],[1,.4]),('tanh+even_sq',lambda t,p:np.c_[np.ones(len(t)),np.tanh(p[0]*t),t*t],[1,.4]),('odd_rat+even_sq',lambda t,p:np.c_[np.ones(len(t)),t/(1+p[0]*t*t),t*t],[1])]:
 def fun(p):
  e=[]
  for k in range(4):
   m=np.isfinite(Y[:,k]); z=np.c_[f(A[m,k],p) if not ('even' in name) else chebvander(A[m,k]/1.8,11), f(A[m,js[k]],p)[:,1:] if 'even' in name else chebvander(A[m,js[k]]/1.8,7)[:,1:]-chebvander(np.array([0.]),7)[:,1:]]; v=Y[m,k]; e.extend(z@np.linalg.lstsq(z,v,rcond=None)[0]-v)
  return np.array(e)
 opt=least_squares(fun,a,bounds=(.00001,15),ftol=1e-12,gtol=1e-12); print(name,opt.x,np.sqrt(np.mean(opt.fun**2)))
