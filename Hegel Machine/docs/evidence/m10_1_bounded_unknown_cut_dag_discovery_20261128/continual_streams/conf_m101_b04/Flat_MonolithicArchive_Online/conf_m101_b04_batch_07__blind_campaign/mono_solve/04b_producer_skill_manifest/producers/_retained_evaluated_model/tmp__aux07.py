import pickle,runpy,numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/batch_solver.py'); fit=m['_ls']; aff=m['_affine']; t=s['train']; x=t['x']; z=m['_calibrate'](s['panel_a'],x); w=m['_panel_b'](s['panel_b'],z)
for key in ('panel_b','panel_c'):
 d=s[key]; u=d['u']; print(key,'range',np.round(u.min(0),2),np.round(u.max(0),2))
 for k,v in d.items():
  if k=='u': continue
  splits=np.array_split(np.random.default_rng(907).permutation(len(u)),4); allidx=np.arange(len(u)); print(k,'output std',np.round(v.std(0),4));
  for name,design in [('affine',aff),('cyclic',lambda a:np.c_[aff(a),a*np.roll(a,-1,1),a**3]),('quadratic',lambda a:np.c_[aff(a),*[np.prod(a[:,j],1) for j in cwr(range(4),2)]]),('tanh',lambda a:np.c_[np.ones(len(a)),np.tanh(a)]),('sine',lambda a:np.c_[np.ones(len(a)),np.sin(a)]),('sinwarp',lambda a:aff(a-.4*np.sin(2.85*a)))]:
   a=design(u); print(name,round(np.mean(abs(a@fit(a,v)-v)),6),np.round([np.mean(abs(a[j]@fit(a[i],v[i])-v[j])) for j in splits for i in [np.setdiff1d(allidx,j)]],5))
 print('first rows',np.round(np.c_[*d.values()][:3],4))
print('time',s['eval_times'],'telemetry',s['telemetry_vec']); print('test vs train quantiles W norm',*[np.round(np.quantile(np.linalg.norm(q-w.mean(0),axis=1),[.5,.8,.95,1]),4) for q in [w,m['_panel_b'](s['panel_b'],m['_calibrate'](s['panel_a'],s['test']['x']))]])
