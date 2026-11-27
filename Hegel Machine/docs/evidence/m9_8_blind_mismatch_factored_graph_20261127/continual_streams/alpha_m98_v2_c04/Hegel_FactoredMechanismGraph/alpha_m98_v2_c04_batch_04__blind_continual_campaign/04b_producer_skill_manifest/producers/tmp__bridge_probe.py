import itertools as it,pickle,numpy as np
np.set_printoptions(precision=5,suppress=True,linewidth=140)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']

def gram(deg):
 return [p for k in range(deg+1) for p in it.combinations_with_replacement(range(4),k)]
def design(t,p):
 return np.column_stack([np.prod(t[:,i],axis=1) for i in p])
def fit(a,b):
 w=np.ones(len(b))
 for k in range(16):
  s=np.sqrt(w[:,None]); C=np.linalg.lstsq(a*s,b*s,rcond=None)[0]; r=np.linalg.norm(b-a@C,axis=1); q=max(np.median(r)*2.5,0.00035); w=1/(1+(r/q)**2)
 good=r<max(np.median(r)*5,0.001)
 if good.sum()>=a.shape[1] and np.linalg.matrix_rank(a[good])==a.shape[1]: C=np.linalg.lstsq(a[good],b[good],rcond=None)[0]
 return C

if __name__=='__main__':
 base=gram(1); quad=gram(2); cubic=gram(3); quart=gram(4)
 cases={'linear':base,'quadratic':quad,'quad+diagonal cubic':quad+[(i,)*3 for i in range(4)],'cubic':cubic,'cubic+diagonal quartic/quintic':cubic+[(i,)*k for k in [4,5] for i in range(4)],'quartic':quart}
 clear=np.linalg.norm(y,axis=1)<12
 print('gross rows:',np.flatnonzero(~clear)); print('sweep fields',[(k,np.shape(v)) for k,v in c['single_channel_sweeps'].items()]); print('operational ranges')
 for key in ('train','test'):
  for field in ('past_channel_raw','hold_channel_raw'):
   a=d[key][field].reshape(-1,4); print(key,field,a.min(0),a.max(0))
 splits=[(f'mixed{i}',np.arange(16+i,80,4)) for i in range(4)]+[(f'axis{i}',np.arange(4*i,4*i+4)) for i in range(4)]
 strength=np.max(np.abs(x),axis=1); splits.append(('outer mixed',np.flatnonzero((strength>1.45)&(np.arange(80)>=16))))
 for name,p in cases.items():
  A=design(x,p); C=fit(A[clear],y[clear]); r=np.linalg.norm(y-A@C,axis=1)
  print(name,'p',len(p),'inliers q50,90,max',np.quantile(r[clear],[.5,.9,1]),'held-out med/max by slice')
  for label,idx in splits:
   test=np.intersect1d(idx,np.flatnonzero(clear)); train=clear.copy(); train[idx]=False
   if len(test)==0 or sum(train)<len(p): continue
   err=np.linalg.norm(y[test]-A[test]@fit(A[train],y[train]),axis=1)
   print(label, np.round([np.median(err),np.max(err)],6),end='; ')
  print()
  if name in ('quad+diagonal cubic','cubic','cubic+diagonal quartic/quintic'):
   for power,co in zip(p,C):
    if np.max(abs(co))>0.001: print(power,co)
