import numpy as np
from scipy.optimize import least_squares
from probe_walls05 import X,Y,B,s,forward,cv,ix
from scipy.optimize import root

def data(j):
 take=B['bank_id']==j; xb=np.zeros((take.sum(),4)); xb[:,0]=B['ref_signal'][take]; x=np.r_[X[:,(j+np.arange(4))%4],xb]; y=np.r_[Y[:,j],B['raw_readout'][take]]; keep=abs(y)<5;return x[keep],y[keep]

def model(x,q,kind):
 a,b,c,d=x.T; bias,amp,k,h,t,c1,c2,v,mix=q[:9]; z=np.maximum(a-t,0); power=kind%4+1
 if kind//4==0:cross=(c1*c+c2*c*c)/(1+v*c*c)
 if kind//4==1:cross=c1*np.tanh(v*c)+c2*c*c
 if kind//4==2:cross=c1*c+c2*c*c/(1+v*c*c)
 if kind//4==3:cross=c1*c/(1+v*c*c)+c2*c*c
 if kind//4==4:cross=(c1*c+c2*c*c)/(1+v*(a*a+c*c))
 if kind//4==5:cross=c1*c+c2*c*c
 if kind//4==6:cross=(c1*c+c2*c*c)/(1+v*(a+c)**2)
 if kind//4==7:cross=(c1*c+c2*c*c)/(1+v*(a-c)**2)
 if kind//4==8:cross=c1*c/(1+v*c*c)+c2*c*c/(1+q[9]*c*c)
 return bias+amp*np.tanh(k*a)+h*z**power+cross+mix*a*c

def fit(x,y,kind):
 q=[0,1.5,1.17,.3,1.2,.3,-.15,.4,0,.4]; lo=[-1,.3,.4,0,.6,0,-1,.00001,-.15,.00001];hi=[1,3,3,20,1.49,2,0,5,.15,5]
 return least_squares(lambda q:model(x,q,kind)-y,q,bounds=(lo,hi),loss='soft_l1',f_scale=.0003,max_nfev=250).x

if __name__=='__main__':
 scores=[]
 for k in range(36):
  qs=[]; stats=[]
  for j in range(4):
   x,y=data(j);q=fit(x,y,k);qs.append(q);err=[]
   for i in range(5):
    test=np.arange(len(y))%5==i; err.extend(model(x[test],fit(x[~test],y[~test],k),k)-y[test]);
   stats.append([*np.quantile(abs(np.array(err)),[.5,.9,.99]),np.median(abs(model(x,q,k)-y))])
  scores.append((np.max(stats,axis=0),k,qs));
 scores.sort(key=lambda a:a[0][1]);
 for err,k,qs in scores[:8]:
  print(k,err,np.round(qs,5));
  def inv(y):return root(lambda t:np.array([model(t[None,(j+np.arange(4))%4],q,k)[0] for j,q in enumerate(qs)])-y,y/1.6).x
  yc=np.array([inv(y) for y in s['y_obs_raw'].reshape(-1,4)]).ravel();S,_=forward((1,1));print('chamber',cv(S,yc,np.array_split(ix,8,axis=1)),cv(S,yc,ix))
