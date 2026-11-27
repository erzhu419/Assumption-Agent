from rheo_probe import *
from scipy.optimize import least_squares
L=g['lab_id']; D=th[2]+xg@th[3:]; E=-np.expm1(-t[:,None]/tau); M=pg@(-np.expm1(-dt[:,None]/tau)*np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0][:,None]/tau)); G=(hg[:,None]-M)*th[:2]; reset=hg[:,None]*th[:2]; y=np.c_[yc,yr]; s=c['single_channel_sweeps']; print('sweep affine true|cmd:',[np.polyfit(s['cmd_stress'][s['lab_id']==l],s['true_stress'][s['lab_id']==l],1) for l in np.unique(L)])
def drift(k,kind):
 if kind=='log':return np.log1p(k*t)/k
 if kind=='sat':return -np.expm1(-k*t)/k
 if kind=='rat':return t/(1+k*t)
 if kind=='sqrt':return 2*t/(np.sqrt(1+k*t)+1)
 return t

def pred(p,kind,mode):
 k=np.exp(p[0]); amp=G; tt=tau; d=D
 if mode=='reset':amp=reset
 if mode=='mix':amp=G+p[1]*(reset-G)
 if mode=='gain':amp=G*np.exp(p[1:3])
 if mode=='tau':tt=tau*np.exp(p[1:3])
 if mode=='drift':d=D*p[1]+p[2]
 if mode=='boundary':amp=(hg[:,None]*p[1]+p[2]-M)*th[:2]
 return np.c_[amp@(-np.expm1(-t[:,None]/tt)).T+d[:,None]*drift(k,kind),amp@(1/tt)+d]
for kind in ('lin','log','sat','rat','sqrt'):
 for mode in ('gap','reset','mix','gain','tau','drift','boundary'):
  if kind=='lin' and mode not in ('gap','reset'):continue
  q={'mix':[0.], 'gain':[0.,0.],'tau':[0.,0.],'drift':[1.,0.],'boundary':[1.,0.]}.get(mode,[]); init=[0.,*q]; err=[]; par=[]
  for l in [None,*np.unique(L)]:
   mask=np.ones(len(L),dtype=bool) if l is None else L!=l
   p=least_squares(lambda p: ((pred(p,kind,mode)-y)[mask]*np.r_[np.ones(len(t)),0.45]).ravel(),init,loss='soft_l1',f_scale=.009,max_nfev=80).x
   if l is None:par=np.r_[np.exp(p[0]),p[1:]]; full=np.sqrt(np.mean((pred(p,kind,mode)-y)**2,axis=0)); continue
   r=(pred(p,kind,mode)-y)[~mask];err.append([np.sqrt(np.mean(r[:,:-1]**2)),np.sqrt(np.mean(r[:,-1]**2))])
  print(kind,mode,np.round(par,4),'full',np.round([np.linalg.norm(full[:-1])/np.sqrt(len(t)),full[-1]],5),'LOLO max',np.round(np.max(err,axis=0),5))
