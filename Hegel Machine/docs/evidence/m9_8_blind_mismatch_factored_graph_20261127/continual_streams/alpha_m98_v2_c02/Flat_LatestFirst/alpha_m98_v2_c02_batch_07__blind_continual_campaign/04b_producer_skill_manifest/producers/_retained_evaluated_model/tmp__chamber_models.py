import numpy as np
from scipy.optimize import least_squares
q=np.load('/tmp/chamber_inputs.npz'); globals().update({k:q[k] for k in q.files})
obs=np.c_[Y,V]; K=np.vstack([kernel,1/T]); lab=np.asarray(lab); tail=np.cumsum(dt[::-1])[::-1]-dt
print('baseline history coefficients',np.round(-W@(H[:2,None]*K.T),4))
def pred(p,mode):
 wait,a,b,g=p[:4]; scale=1.; fac=np.ones(2); s=1.; last=0.
 if mode=='clock': scale=p[4]
 if mode=='branch': fac=p[4:6]
 if mode=='hold': s=p[4]
 if mode=='last': last=p[4]
 if mode=='branchclock': fac=np.array(p[4:6]); scale=p[6]
 if mode=='splitclock': scale=np.asarray(p[4:6])
 if mode=='clocklast': scale,last=p[4:6]
 w=-np.expm1(-dt[:,None]*scale/T)*np.exp(-(tail[:,None]*scale+wait)/T)
 z1=(past@w)*fac+last*past[:,-1,None]*(1-np.exp(-dt[-1]/T))
 return ((s*hold[:,None]-z1)*H[:2])@K.T + drift[:,None]*np.r_[a*t-b*g*np.expm1(-t/g),a+b]
settings={'gate':([],[] ,[]),'clock':([1.],[.25],[4.]),'branch':([1.,1.],[.2,.2],[2.,2.]),'hold':([1.],[.8],[1.2]),'last':([0.],[-.6],[.6]),'branchclock':([1.,1.,1.],[.2,.2,.25],[2.,2.,4.]),'splitclock':([1.,1.],[.25,.25],[4.,4.]),'clocklast':([1.,0.],[.25,-.6],[4.,.6])}
for mode,(x,lo,hi) in settings.items():
 def fit(idx):
  return least_squares(lambda p:(pred(p,mode)[idx]-obs[idx]).ravel(),[0.,0.,1.,1.]+x,bounds=([-1,-.4,.2,.2]+lo,[3,.6,2.,4.]+hi),loss='soft_l1',f_scale=.009).x
 sol=fit(np.arange(len(Y))); r=pred(sol,mode)-obs
 oof=np.empty_like(obs)
 for l in np.unique(lab):
  idx=lab==l; oof[idx]=pred(fit(~idx),mode)[idx]-obs[idx]
 print(mode,np.round(sol,5),'fit',np.sqrt(np.mean(r**2,axis=0))[[0,5,11,12]],'OOF worst',max(np.sqrt(np.mean(oof[lab==l,:-1]**2)) for l in np.unique(lab)),max(np.sqrt(np.mean(oof[lab==l,-1]**2)) for l in np.unique(lab)))
