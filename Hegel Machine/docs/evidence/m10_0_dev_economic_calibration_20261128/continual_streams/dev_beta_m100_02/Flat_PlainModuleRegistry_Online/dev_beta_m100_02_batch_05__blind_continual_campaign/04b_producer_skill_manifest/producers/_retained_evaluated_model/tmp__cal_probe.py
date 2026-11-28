import itertools,json,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; z=s['y_obs_raw'].reshape(-1,4)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
print('shape',x.shape,z.shape,s['y_obs_raw'].shape,'pulses',len(s.get('reference_tracer_pulses',[])))
print('cal first/mid/last',np.c_[x,y][np.r_[0:9,37:42,74:80]])
print('x singular',np.linalg.svd(x-x.mean(0),compute_uv=False))
print('bank type/shape',type(c['single_bank_sweeps']),[(k,np.shape(v)) for k,v in c['single_bank_sweeps'].items()] if isinstance(c['single_bank_sweeps'],dict) else [(type(a),[(k,np.shape(v)) for k,v in a.items()]) for a in c['single_bank_sweeps'][:2]])
print('observations first',z[:8]); print('aux',type(s['aux_manifold_telemetry']),np.shape(s['aux_manifold_telemetry']))
for fn in ['public_asset_index.json','/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json','/tmp/hegel_continual_archive/archive_manifest.json']:
 a=json.load(open(fn)); print(fn,str(a)[:1200])

def basis(x,m):
 a=[np.ones(len(x)),*x.T]
 if m=='full2': a += [x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]
 if m=='full3': a += [np.prod(x[:,idx],axis=1) for q in (2,3) for idx in itertools.combinations_with_replacement(range(4),q)]
 if m=='qcross_cubic': a += [x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]+list((x**3).T)
 if m=='cubic_cross': a += [x[:,i]*x[:,j] for i,j in itertools.combinations(range(4),2)]+list((x**3).T)
 if m=='cubic': a += list((x**3).T)
 if m=='quad_cubic': a += list((x**2).T)+list((x**3).T)
 if m=='cubic5': a += list((x**3).T)+list((x**5).T)
 if m.startswith('tanh'): a += list(np.tanh(float(m[4:])*x).T)
 return np.column_stack(a)

def fit(a,y):
 w=np.ones(len(a))
 for _ in range(25):
  p=np.linalg.lstsq(a*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=1e-10)[0]; r=np.linalg.norm(y-a@p,axis=1); w=np.minimum(1, max(np.median(r)*1.5,1e-7)/np.maximum(r,1e-12))
 keep=r<max(8*np.median(r),.03)
 return np.linalg.lstsq(a[keep],y[keep],rcond=1e-10)[0]

for m in ['affine','full2','cubic','quad_cubic','cubic_cross','qcross_cubic','cubic5','tanh0.4','tanh0.8','tanh1.2','full3']:
 a=basis(x,m); p=fit(a,y); r=np.linalg.norm(a@p-y,axis=1); print(m,'condition',round(np.linalg.cond(a),1),'fit p50/p90',np.quantile(r,[.5,.9]),'spikes',np.flatnonzero(r>2))
 for order in (np.arange(len(x)),np.random.default_rng(11).permutation(len(x)),np.argsort(np.linalg.norm(x,axis=1))):
  scores=[]
  for test in np.array_split(order,5):
   train=np.setdiff1d(np.arange(len(x)),test); rt=np.linalg.norm(a[test]@fit(a[train],y[train])-y[test],axis=1); rt=rt[r[test]<2]; scores.append(np.sqrt(np.mean(rt**2)))
  print(' worst',round(max(scores),6),end='')
 print()
 if m in ['affine','cubic_cross','qcross_cubic','tanh0.8']: print(' coeff',p.T)
