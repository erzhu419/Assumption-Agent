import itertools, pickle
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
b=c['single_bank_sweeps']; bid=b['bank_id']; bx=np.eye(4)[bid]*b['ref_signal'][:,None]; by=b['raw_readout']
all_terms=[k for d in (2,3) for k in itertools.combinations_with_replacement(range(4),d)]
def terms(name,j):
 if name=='cyclic': return [(j,(j+1)%4),(j,j,j)]
 if name=='own': return [(j,k) for k in range(4)]+[(j,j,j)]
 if name=='paircubic': return [k for k in all_terms if len(k)==2 or len(set(k))==1]
 return all_terms
def mat(x,name,j): return np.column_stack([np.ones(len(x)), x, *[x[:,t].prod(1) for t in terms(name,j)]])
def fit(name,tr,bt):
 A=[mat(x[tr],name,j) for j in range(4)]; B=[mat(bx[bt & (bid==j)],name,j) for j in range(4)]; T=[by[bt & (bid==j)] for j in range(4)]
 w=np.ones(len(tr)); v=[np.ones(len(t)) for t in T]
 for it in range(18):
  p=[np.linalg.lstsq(np.vstack((a*w[:,None],bb*u[:,None])),np.r_[y[tr,j]*w,t*u],rcond=None)[0] for j,(a,bb,t,u) in enumerate(zip(A,B,T,v))]
  e=np.linalg.norm(np.column_stack([a@q for a,q in zip(A,p)])-y[tr],axis=1); sig=max(np.median(e)/1.83,1e-8)
  w=1/np.hypot(1,e/(4*sig)); v=[1/np.hypot(1,(bb@q-t)/(3*sig)) for bb,q,t in zip(B,p,T)]
 return p
def pred(x,name,p): return np.column_stack([mat(x,name,j)@p[j] for j in range(4)])
N=len(x); allr=np.arange(N); tests={}
for j,test in enumerate(np.array_split(np.random.default_rng(31).permutation(N),5)): tests[f'random{j}']=test
for j in range(4): tests[f'axis{j}']=np.flatnonzero((np.count_nonzero(x,axis=1)==1)&(x[:,j]!=0))
tests['edge']=np.argsort(np.max(np.abs(x),axis=1))[-16:]; tests['interior']=np.argsort(np.linalg.norm(x,axis=1))[:16]
for name in ['cyclic','own','paircubic','full']:
 print(name)
 for use_bank in [False,True]:
  stat={}; spikes=[]
  for label,test in tests.items():
   bt=np.ones(len(bid),bool) if use_bank else np.zeros(len(bid),bool)
   if label.startswith('axis'): bt &= bid!=int(label[-1])
   if label=='edge': bt &= np.abs(b['ref_signal'])<np.min(np.max(np.abs(x[test]),axis=1))
   p=fit(name,np.setdiff1d(allr,test),bt); e=np.linalg.norm(pred(x[test],name,p)-y[test],axis=1); clean=~np.isin(test,[13,14,15]); stat[label]=float(np.max(e[clean],initial=0)); spikes.extend(e[~clean])
  p=fit(name,allr,np.full(len(bid),use_bank)); er=np.linalg.norm(pred(x,name,p)-y,axis=1)
  print('bank',use_bank,'max by slice', {k:round(v,8) for k,v in stat.items()},'min held-out spike',min(spikes),'clean max fit',np.max(np.sort(er)[:-3]))
  if name=='cyclic':
   print('ratios',np.array([[q[1+(j+1)%4]/q[1+j],q[1+(j-1)%4]/q[1+j],q[-2]/q[1+j],q[-1]/q[1+j]] for j,q in enumerate(p)]))
