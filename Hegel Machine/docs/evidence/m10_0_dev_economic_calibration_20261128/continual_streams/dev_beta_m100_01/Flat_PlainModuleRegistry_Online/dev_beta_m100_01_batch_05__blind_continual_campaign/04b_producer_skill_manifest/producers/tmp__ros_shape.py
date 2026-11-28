import numpy as np
from scipy.optimize import least_squares
from ros_probe import x,y
from ros_joint_input import bank
bi=bank['bank_id']; z=bank['ref_signal']; sy=bank['raw_readout']; keep=np.max(np.abs(y),axis=1)<5; sb=np.abs(sy)<5; opp=[2,3,0,1]; axis=np.count_nonzero(x,axis=1)==1

def features(a,k,m):
 t=k[0]*a
 if m=='arg3': t=t+k[2]*a**3
 if m=='argsin': t=t+k[2]*np.sin(k[3]*a)
 if m=='argrat': t=t+k[2]*a/(1+k[3]*a*a)
 d=np.tanh(t)+k[1]*a
 if m=='sin': d+=k[2]*np.sin(k[3]*a)
 if m=='sin3': d+=k[2]*np.sin(k[3]*a)+k[4]*a**3
 if m=='cos_even': d+=k[2]*(1-np.cos(k[3]*a))
 if m=='tanh2': d+=k[2]*np.tanh(k[3]*a)
 if m=='rat': d+=k[2]*a/(1+k[3]*a*a)
 return d

def local(p,a,m): return p[:4]+p[4:8]*features(a,p[8:],m)
if __name__=='__main__':
 for m in ('aff','arg3','argsin','argrat','sin','sin3','cos_even','tanh2','rat'):
  n={'aff':2,'arg3':3,'sin3':5}.get(m,4)
  def fun(p):
   ss=p[:4][bi]+p[4:8][bi]*features(z,p[8:],m); a=local(p,x,m); on=(x!=0)&(keep&axis)[:,None]; return np.r_[(ss-sy)[sb],(a-y)[on]]
  lo=[-1]*4+[.1]*4+[.2,-1,-3,.1,-1][:n]; hi=[1]*4+[5]*4+[4,2,3,8,1][:n]; best=min((least_squares(fun,[.07,-.06,.08,-.05,1.4,.9,1.4,.9]+[1.35,.18,extra,k,0.][:n],bounds=(lo,hi),max_nfev=350) for k in (1.2,2.5,4.,6.) for extra in (-.1,.1)),key=lambda q:q.cost); p=best.x
  r=y-local(p,x,m); a=x[:,opp]; tr=keep&axis; ds=[]; cs=[]
  for f in ('cos','rad','tanh'):
   def base(k):
    even=1-np.cos(k*a) if f=='cos' else (a*a/(1+k*a*a) if f=='rad' else np.tanh(k*a)**2); return a,even
   def reg(k):
    l,e=base(k); pred=np.empty_like(r); b=[]
    for j in range(4):
     pick=tr&(a[:,j]!=0); D=np.c_[l[:,j],e[:,j]]; bj=np.linalg.lstsq(D[pick],r[pick,j],rcond=None)[0];pred[:,j]=D@bj;b.append(bj)
    return pred,np.asarray(b)
   k=least_squares(lambda k:(reg(k)[0]-r)[tr&(np.any(a!=0,axis=1))].ravel(),[1.7],bounds=(.1,5)).x; rr=r-reg(k)[0]; ds.append((f,float(k[0]),float(np.sqrt(np.mean(rr[keep&~axis]**2)))));cs.append(reg(k)[1])
  print(m,'axis RMSE',np.sqrt(np.mean(best.fun**2)),'p',p.round(6),'mixed',ds,flush=True)
  if m in ('arg3','argsin','sin','sin3'): print('coupling',cs[0].round(6),flush=True)
 print('alternating-pair mixed refs/raw',np.c_[x[16:26],y[16:26]].round(5))
