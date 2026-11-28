import numpy as np
from scipy.optimize import least_squares
from itertools import combinations

def experiment(s):
 c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; q=c['single_bank_sweeps']; clean=np.max(abs(y),axis=1)<5; isolated=np.count_nonzero(x,axis=1)==1
 def shape(t,k):return np.tanh(k*t)
 rates=[]
 for j in range(4):
  m=(q['bank_id']==j)&(abs(q['raw_readout'])<5); a=q['ref_signal'][m]; b=q['raw_readout'][m]; fit=least_squares(lambda p:p[0]+p[1]*shape(a,p[3])+p[2]*a-b,[0,1,0,1.4],bounds=([-1,-10,-5,.1],[1,10,5,8]),gtol=1e-14,xtol=1e-14,ftol=1e-14);print('own',j,fit.x,'rms',np.sqrt(np.mean(fit.fun**2))); rates.append(fit.x[-1])
 for j in range(4):
  for k in range(4):
   m=clean&isolated&(x[:,k]!=0); a=x[m,k]; b=y[m,j]; print('isolated',j,k,np.round(b,5))
 rates=np.array(rates); print('aux head',s['aux_manifold_telemetry'][:5]); print('bank',s['row_bank_id'][:40])
 u=x[clean]; cy=y[clean]; v=shape(u,rates); base=np.c_[np.ones(len(u)),u,v]; cols=[]
 for i,j in combinations(range(4),2):
  for n,a,b in [('xx',u[:,i],u[:,j]),('xt',u[:,i],v[:,j]),('tx',v[:,i],u[:,j]),('tt',v[:,i],v[:,j])]:cols.append((f'{n}{i}{j}',a*b))
 for j in range(4):
  b=cy[:,j]; A=base.copy(); chosen=[]
  for _ in range(8):
   p=np.linalg.lstsq(A,b,rcond=None)[0]; e=b-A@p; print('mixed',j,chosen,'rms',np.sqrt(np.mean(e**2)))
   if np.sqrt(np.mean(e**2))<3e-5:break
   opts=[]
   for name,col in cols:
    if name not in chosen:
     B=np.c_[A,col]; opts.append((np.linalg.norm(b-B@np.linalg.lstsq(B,b,rcond=None)[0]),name,col))
   _,name,col=min(opts,key=lambda v:v[0]); chosen.append(name); A=np.c_[A,col]
  print('coef',np.round(p,6))
if __name__=='__main__':
 import pickle
 experiment(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'])
