import itertools as it
import json, pickle
import numpy as np
from pathlib import Path
p=Path('/holdout_task')
d=pickle.loads((p/'public_batch.pkl').read_bytes())[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; axis=(np.count_nonzero(x,axis=1)==1)
for a in json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text()):
 t=np.asarray(a['facility_telemetry_trace']); print(a['completed_campaign_short_name'],'median',np.median(t,0).round(3),'sd',np.std(t,0).round(3))
print('current auxiliary median',np.median(d['train']['aux_chamber_telemetry'],0).round(3))
def powers(pred):
 return [ids for k in range(1,5) for ids in it.combinations_with_replacement(range(4),k) if pred(ids)]
def features(x, terms):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ids],axis=1) for ids in terms])
def fit(A,Y):
 b=np.linalg.lstsq(A,Y,rcond=None)[0]
 for _ in range(15):
  r=np.linalg.norm(Y-A@b,axis=1); s=max(np.median(r)*2,.00001); w=1/(1+(r/s)**2)**.5
  b=np.linalg.lstsq(A*w[:,None],Y*w[:,None],rcond=None)[0]
 return b
base=features(x,powers(lambda s:len(s)==1)); e=np.linalg.norm(y-base@fit(base,y),axis=1); print('gross rows',[(int(i),round(e[i],3)) for i in np.flatnonzero(e>4)])
rng=np.random.default_rng(834); cats=[]
for named, keep in [('affine',lambda s:len(s)==1),('own poly 2',lambda s:len(s)<=2 and len(set(s))==1),('quadratic',lambda s:len(s)<=2),('quadratic + own cubic',lambda s:len(s)<=2 or (len(s)==3 and len(set(s))==1)),('quad + own cubic quartic',lambda s:len(s)<=2 or len(set(s))==1),('cubic',lambda s:len(s)<=3),('cubic + own quartic',lambda s:len(s)<=3 or len(set(s))==1)]:
 terms=powers(keep); A=features(x,terms); ef=[]
 for rep in range(6):
  order=np.r_[rng.permutation(np.flatnonzero(~axis)),rng.permutation(np.flatnonzero(axis))]; fold=np.empty(len(x),int); fold[order]=np.arange(len(x))%5
  er=np.zeros_like(y)
  for k in range(5):
   i=fold!=k; er[~i]=A[~i]@fit(A[i],y[i])-y[~i]
  ef.append(er)
 rng=np.random.default_rng(834)
 ef=np.array(ef); power=np.max(np.abs(x),axis=1); mask=e<4
 print(named,'clean / axis / mixed edge RMS',*[round(float(np.sqrt(np.mean(ef[:,m]**2))),5) for m in [mask,mask & axis,mask & ~axis & (power>1.3)]], 'sum RMS',round(float(np.sqrt(np.mean(ef[:,mask].sum(-1)**2))),5))
 if named in ('quadratic + own cubic','cubic'):
  B=fit(A,y); print('terms',[(s,np.round(b,3).tolist()) for s,b in zip([()]+terms,B)]); print('inlier quantiles',np.quantile(np.linalg.norm(y-A@B,axis=1)[mask],[0,.5,.9,1]).round(6))
