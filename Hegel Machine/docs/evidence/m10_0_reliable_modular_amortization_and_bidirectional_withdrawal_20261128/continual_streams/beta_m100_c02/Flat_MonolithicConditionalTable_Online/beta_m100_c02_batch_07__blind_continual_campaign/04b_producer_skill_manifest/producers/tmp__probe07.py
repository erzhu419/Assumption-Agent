import pickle,numpy as np,itertools,importlib.util
from scipy.optimize import least_squares
from pathlib import Path
P=Path('/tmp/hegel_continual_archive'); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']; q=c['single_bank_sweeps']; j=q['bank_id']; xs=np.eye(4)[j]*q['ref_signal'][:,None]; vs=q['raw_readout']; W=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]); YP=np.array([p['y_pulse_raw'] for p in item['campaign_diagnostics']]).reshape(3,-1,4)
def load(p):
 z=importlib.util.spec_from_file_location(p.stem,p); m=importlib.util.module_from_spec(z);z.loader.exec_module(m);return m
mods={p.stem.split('_')[-2]:load(p) for p in sorted(P.glob('*solver.py'))}
base=mods['06']; per=mods['04'];

def additive(x,d):return np.column_stack([np.ones(len(x))]+[x[:,i]**p for i in range(4) for p in range(1,d+1)])
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,inds],axis=1) for p in range(1,d+1) for inds in itertools.combinations_with_replacement(range(4),p)])

def fit(fn,train):
 b=[]
 for k in range(4):
  a=np.r_[fn(X[train]),fn(xs[j==k])]; y=np.r_[Y[train,k],vs[j==k]]
  b.append(np.linalg.lstsq(a,y,rcond=None)[0]);
 return np.array(b).T
for kind in ('add','poly'):
 for d in range(1,8 if kind=='add' else 5):
  fn=lambda x: (additive if kind=='add' else poly)(x,d)
  errs=[]
  for k in range(5):
   mask=np.arange(len(X))%5==k; errs.extend(fn(X[mask])@fit(fn,~mask)-Y[mask]);
  errs=np.array(errs);print(kind,d,'CV RMS',np.sqrt(np.mean(errs**2,axis=0)), 'worst',np.max(abs(errs)))
  if kind=='add' and d in (3,5): print(np.round(fit(fn,np.ones(len(X),bool)).T,5))
for k in ('04','05','06'):
 m=mods[k];
 try:
  model=m.calibration_map(c); pred=m.raw_map(X,*model) if k=='05' else m.raw_map(X,model);print('archive fit',k,np.sqrt(np.mean((Y-pred)**2,axis=0)))
 except Exception as e:print(k,e)
fn=lambda x:additive(x,5); B=fit(fn,np.ones(len(X),bool));
def rpt(name,S):
 y=(W@S.T).reshape(-1,4); e=(fn(y)@B).reshape(YP.shape)-YP; print(name, np.sqrt(np.mean(e**2,axis=(1,2))), 'max',np.max(abs(e)), 'pred range', (y.min(),y.max()))
rpt('Robin',base.chamber_response(s)[0]);rpt('sealed',base.chamber_response(s,np.zeros(4))[0]);rpt('periodic x',per.chamber_response(s)[0]);
for k in ('01','02','03','05'):
 m=mods[k]
 try:rpt(k,m.chamber_response(s)[0]);
 except Exception as e:print(k,e)
