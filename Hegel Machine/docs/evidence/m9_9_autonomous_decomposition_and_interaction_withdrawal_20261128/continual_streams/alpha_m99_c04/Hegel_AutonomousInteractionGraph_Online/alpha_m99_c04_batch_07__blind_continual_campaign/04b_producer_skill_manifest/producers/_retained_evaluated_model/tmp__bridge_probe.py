import json,pickle,itertools,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
for e in json.load(open(p)):
 print('log',e.get('completed_campaign_index'),[(k, str(v)[:420]) for k,v in e.items() if any(s in k.lower() for s in ('bridge','transduc','setting','campaign_id'))], list(e))
d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; np.set_printoptions(precision=5,suppress=True)
print('axis rows raw',y[:16]); print('axis loads', x[:16])
for name,terms in [('affine',[()]+[(i,) for i in range(4)]),('diagonal cubic',[()]+[(i,) for i in range(4)]+[(i,i,i) for i in range(4)]),('diagonal cube plus quadratic',[()]+[(i,) for i in range(4)]+[(i,i,i) for i in range(4)]+list(itertools.combinations_with_replacement(range(4),2))),('full cubic',[()]+[t for k in (1,2,3) for t in itertools.combinations_with_replacement(range(4),k)])]:
 A=np.column_stack([x[:,t].prod(1) for t in terms]); err=np.zeros_like(y); B=[]
 for fold in range(5):
  tr=np.arange(len(y))%5!=fold; U=A[tr]; V=y[tr]; b=np.linalg.lstsq(U,V,rcond=None)[0]
  for eps in (1., .12, .02, .004, .001):
   for _ in range(25):
    w=1/np.maximum(np.linalg.norm(U@b-V,axis=1),eps); b=np.linalg.lstsq(U*np.sqrt(w[:,None]),V*np.sqrt(w[:,None]),rcond=None)[0]
  B.append(b); err[~tr]=A[~tr]@b-y[~tr]
 r=np.linalg.norm(err,axis=1); order=np.argsort(r)
 print(name,'norm quantiles',np.quantile(r,[.5,.8,.9,.95,1]),'largest',list(zip(order[-9:],r[order[-9:]].round(5)))); print('terms',list(zip(terms,np.median(B,axis=0).round(5).tolist())))
