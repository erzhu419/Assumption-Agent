from cal_sat import X,Y,ids,sig,z
import numpy as np
from scipy.optimize import least_squares

def self_curve(u,p,kind):
 a,b,c,d=p
 if kind=='rat3': return a*u/(1+b*u*u)+c*u+d*u**3
 if kind=='tanh3': return a*np.tanh(b*u)+c*u+d*u**3
 if kind=='soft3': return a*u/np.sqrt(1+b*u*u)+c*u+d*u**3
 if kind=='inner3': return a*np.tanh(b*u+c*u**3)+d*u
 if kind=='exp3': return a*u*np.exp(-b*u*u)+c*u+d*u**3
 if kind=='rat4': return (a*u+c*u**3)/(1+b*u*u+d*u**4)

def cross_curve(v,p,kind):
 l,q,r,t=p
 if kind=='even': return l*v+q*v*v/(1+r*v*v)
 if kind=='both': return (l*v+q*v*v)/(1+r*v*v)
 if kind=='odd': return l*v/(1+r*v*v)+q*v*v
 if kind=='split': return l*v/(1+r*v*v)+q*v*v/(1+t*v*v)

def data(k):
 sel=ids==k; m=np.zeros((sum(sel),4)); m[:,k]=sig[sel]
 x=np.r_[X,m]; y=np.r_[Y[:,k],z[sel]]; keep=abs(y)<5
 return x[keep],y[keep],np.flatnonzero(keep)

def predict(p,x,k,sk,ck):
 u=x[:,k]; v=x[:,(k+2)%4]
 a=x[:,(k+1)%4]; c=x[:,(k+3)%4]
 return p[0]+self_curve(u,p[1:5],sk)+cross_curve(v,p[5:9],ck)+p[9]*u*a+p[10]*u*c+p[11]*a*c+p[12]*a+p[13]*c

def fit(x,y,k,sk,ck):
 lo=np.full(14,-np.inf); hi=np.full(14,np.inf); lo[[1,2,7,8]]=0
 if sk=='rat4': lo[4]=0
 p=np.array([0,1.,1.,0.2,.005,-.2,.1,.3,.3,0,0,0,0,0]); return least_squares(lambda p:predict(p,x,k,sk,ck)-y,p,bounds=(lo,hi),loss='soft_l1',f_scale=.001,max_nfev=130).x

if __name__=='__main__':
 for sk in ('rat3','tanh3','soft3','inner3','exp3','rat4'):
  for ck in ('even','both','odd','split'):
   errs=[]; params=[]
   for k in range(4):
    x,y,idx=data(k); err=np.empty(len(y)); params.append(fit(x,y,k,sk,ck));
    for f in range(4):
     tr=idx%4!=f; err[~tr]=predict(fit(x[tr],y[tr],k,sk,ck),x[~tr],k,sk,ck)-y[~tr]
    errs.append(np.quantile(abs(err),[.5,.9,1]))
   print(sk,ck,'worst med/q90/max',np.round(np.max(errs,axis=0),6),'ch2',np.round(params[2],5),flush=True)
