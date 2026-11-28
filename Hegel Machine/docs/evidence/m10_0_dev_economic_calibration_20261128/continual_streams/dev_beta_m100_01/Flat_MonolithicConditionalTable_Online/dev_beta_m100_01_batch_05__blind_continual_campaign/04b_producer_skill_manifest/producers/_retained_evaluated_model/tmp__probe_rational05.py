import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; X,Y=sw['ref_rosette'],sw['raw_rosette']; B=sw['single_bank_sweeps']; print('nonzero inputs',np.count_nonzero(X,axis=1)); print(np.round(np.c_[X[16:32],Y[16:32]],4))
def poly(X,n):
 return np.column_stack([np.prod(X[:,t],axis=1) for k in range(n+1) for t in combinations_with_replacement(range(X.shape[1]),k)])
def spec(x,kind):
 a,b,c,d=x.T; ac=x[:,[0,2]]; P=poly(ac,3); C=np.c_[np.ones(len(a)),a*a,c*c,a*c]; D=np.c_[a*a,c*c,a*c,b*b,d*d];
 if kind==0:return np.c_[P,b,d],D[:,:3]
 if kind==1:return poly(x,3),D
 if kind in [2,3,4]:return np.c_[P,b,d,a*b,c*b,a*d,c*d,a**5,c**5],np.c_[D[:,:3],a**4,c**4][:,:(3 if kind==2 else 5)]
 return poly(x,3),np.c_[D,a**4,c**4,a**3*c,a*c**3,a*a*c*c]
def model(x,p,kind):
 a,b,c,d=x.T
 if kind==6:
  f=lambda z,k,l: z*(1+l*z*z)/(1+k*z*z)
  return p[0]+p[1]*f(a,p[2],p[3])+p[4]*f(c,p[5],p[6])+p[7]*c*c/(1+p[8]*c*c)+p[9]*a*c+p[10]*b+p[11]*d
 if kind==7 or kind==8:
  t=p[2]*a+p[3]*c+p[4]*a*c+p[5]*c*c
  return p[0]+p[1]*np.tanh(t)+p[6]*a+p[7]*c+p[8]*c*c+p[9]*a*c+p[10]*b+p[11]*d+(p[12]*np.tanh(p[13]*c) if kind==8 else 0)
 if kind in (9,10,11):
  P=poly(x[:,[0,2]],3); K=P.shape[1]; t=p[K]*a+p[K+1]*c+p[K+2]*a*c+p[K+3]*c*c
  out=P@p[:K]+p[K+4]*np.tanh(t)+p[K+5]*np.tanh(p[K+6]*c)+p[K+7]*b+p[K+8]*d
  if kind>=10:out+=p[K+9]*a*a*c*c
  if kind==11:out+=p[K+10]*np.tanh(p[K+11]*a)
  return out
 N,D=spec(x,kind);k=N.shape[1];return (N@p[:k])/(1+D@p[k:])
def fit(x,y,k):
 if k<6:
  N,D=spec(x,k); p=np.linalg.lstsq(np.c_[N,-y[:,None]*D],y,rcond=1e-9)[0];
  if np.min(1+D@p[len(N.T):])<.3:p=np.r_[np.linalg.lstsq(N,y,rcond=1e-9)[0],np.zeros(D.shape[1])]
 else:
  p={6:[0,2,.5,.05,.3,.5,.05,-.1,.5,0,0,0],7:[0,1.7,1.15,.1,0,0,0,.3,-.1,0,0,0],8:[0,1.7,1.15,0,0,0,0,0,-.1,0,0,0,.3,1.],9:[0]*10+[1.15,0,0,0,1.7,.3,1.,0,0],10:[0]*10+[1.15,0,0,0,1.7,.3,1.,0,0,0],11:[0]*10+[1.15,0,0,0,1.7,.3,1.,0,0,0,.03,2]}[k]
 return least_squares(lambda p:model(x,p,k)-y,p,loss='soft_l1',f_scale=.001,max_nfev=550).x
if __name__=='__main__':
 for j in range(4):
  take=B['bank_id']==j; xb=np.zeros((take.sum(),4));xb[:,0]=B['ref_signal'][take];x=np.r_[X[:,(j+np.arange(4))%4],xb];y=np.r_[Y[:,j],B['raw_readout'][take]];ok=abs(y)<5
  for k in (0,1,2,3,5,6,7,8,9,10,11):
   err=[]
   for i in range(5):
    t=(np.arange(len(y))%5==i)&ok;q=fit(x[ok&~t],y[ok&~t],k);err.extend(model(x[t],q,k)-y[t]);
   p=fit(x[ok],y[ok],k); print(j,k,np.round(np.quantile(abs(np.array(err)),[.5,.9,.99]),6),np.round(np.quantile(abs(model(x[ok],p,k)-y[ok]),[.5,.9]),6),np.round(p,4) if k in (6,7,8,9,10,11) else '',flush=True)
