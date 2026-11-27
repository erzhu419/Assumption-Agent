import pickle,numpy as np
from itertools import combinations_with_replacement
with open('/holdout_task/public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
x,y=(d['calibration_load_cells'][k] for k in ('true_channel_loads','bridge_raw'))
clean=np.abs(y).max(1)<8
axis=np.count_nonzero(x,axis=1)==1
print('gross rows',np.flatnonzero(~clean).tolist())
for deg,name in [(1,'affine'),(2,'quadratic'),(3,'diagonal cubic'),(3,'quadratic + diagonal cubic'),(3,'full cubic'),(4,'full quartic')]:
 terms=[t for j in range(deg+1) for t in combinations_with_replacement(range(4),j) if not(name=='diagonal cubic' and len(t)>1 and len(set(t))>1) and not(name=='quadratic + diagonal cubic' and len(t)==3 and len(set(t))>1)]
 a=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]); c=np.linalg.lstsq(a[clean],y[clean],rcond=None)[0]
 e=np.zeros_like(y); ranks=[]
 for fold in range(8):
  held=np.arange(len(x))%8==fold; tr=clean & ~held; te=clean & held
  cc,_,rank,_=np.linalg.lstsq(a[tr],y[tr],rcond=None); e[te]=a[te]@cc-y[te]; ranks.append(rank)
 print(name,len(terms),min(ranks),'OOF RMSE',np.sqrt(np.mean(e[clean]**2,axis=0)).round(7),'max axis/mixed',[round(abs(e[clean & s]).max(),7) for s in (axis,~axis)])
 if name=='full cubic':
  for t,b in zip(terms,c):
   if np.max(abs(b))>0.001: print(t,np.round(b,6))
 for mode in ('train','test'):
  if name=='affine': print(mode,'estimate envelope',[(k,np.round(((d[mode][k]-c[0])@np.linalg.pinv(c[1:])).reshape(-1,4).min(0),2),np.round(((d[mode][k]-c[0])@np.linalg.pinv(c[1:])).reshape(-1,4).max(0),2)) for k in ('past_channel_raw','hold_channel_raw')])
