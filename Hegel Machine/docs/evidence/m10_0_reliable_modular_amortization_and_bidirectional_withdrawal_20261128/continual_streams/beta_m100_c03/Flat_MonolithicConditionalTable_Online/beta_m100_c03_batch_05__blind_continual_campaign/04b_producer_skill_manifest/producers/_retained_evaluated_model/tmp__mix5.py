import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations, combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; q=c['single_bank_sweeps']; keep=np.max(abs(y),axis=1)<5; x,y=x[keep],y[keep]; iso=np.count_nonzero(x,axis=1)==1

def curves():
 def obj(p):
  z=[]
  for j in range(4):
   m=(q['bank_id']==j)&(abs(q['raw_readout'])<5); a=q['ref_signal'][m]; b=q['raw_readout'][m]; z.extend(p[1+3*j]+p[2+3*j]*a+p[3+3*j]*np.sin(p[0]*a)-b)
  return z
 return least_squares(obj,[2.5]+[0,1,.25]*4,gtol=1e-13).x
p=curves(); print('curves',p); f=p[0]; u=x; v=np.sin(f*u); e=1-np.cos(f*u); b=y-p[1::3]-u*p[2::3]-v*p[3::3];
for j in range(4):
 k=(j+2)%4; a=u[:,k]; t=v[:,k]; z=e[:,k]; print('isolated',j,np.c_[a[iso&(a!=0)],b[iso&(a!=0),j]].T)
 dic={'a':a,'v':t,'even':z,'a2':a*a,'a3':a**3,'a4':a**4,'sinhalf':np.sin(.5*f*a),'coshalf':1-np.cos(.5*f*a),'rat':a/(1+a*a),'rat2':a*a/(1+a*a),'abs':abs(a)}
 for i,l in combinations(range(4),2):
  for m,h in [('x',u),('s',v),('c',e)]:
   for n,g in [('x',u),('s',v),('c',e)]:dic[f'{m}{i}{n}{l}']=h[:,i]*g[:,l]
 for i,l,r in combinations(range(4),3):dic[f'x{i}x{l}x{r}']=u[:,i]*u[:,l]*u[:,r]
 target=b[:,j]; opts=[]
 for m,n in combinations(list(dic)[:11],2):
  A=np.c_[dic[m],dic[n]]; sel=iso&(a!=0); co=np.linalg.lstsq(A[sel],target[sel],rcond=None)[0]; opts.append((np.linalg.norm(A[sel]@co-target[sel]),m,n,co))
 print('isolated best',sorted(opts,key=lambda t:t[0])[:5]); sel=[]; A=np.empty((len(u),0));
 for _ in range(7):
  choices=[]
  for name,col in dic.items():
   if name in sel:continue
   B=np.c_[A,col]; co=np.linalg.lstsq(B,target,rcond=None)[0]; choices.append((np.sqrt(np.mean((B@co-target)**2)),name,co,B))
  rms,name,co,A=min(choices,key=lambda t:t[0]); sel.append(name); print('mixed',j,sel,rms,co.round(5));
 print('raw residual quantiles',np.quantile(target,[0,.1,.5,.9,1]))
