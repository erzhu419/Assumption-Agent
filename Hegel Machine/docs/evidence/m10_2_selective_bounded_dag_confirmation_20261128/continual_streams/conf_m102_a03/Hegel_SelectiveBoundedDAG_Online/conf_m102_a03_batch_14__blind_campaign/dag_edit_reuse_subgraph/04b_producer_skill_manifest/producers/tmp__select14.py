import numpy as np,pickle,runpy,itertools
from pathlib import Path
P=Path('/holdout_task'); mod=runpy.run_path('/tmp/trialmod14.py'); fit=runpy.run_path(str(P/'_frozen_nodes/node_01.py'))['_fit']; study=pickle.load(open(P/'public_batch.pkl','rb'))[0]['study']; t=study['train'];
params=[((0,),-5.5,.035,1),((0,),-8,.012,1),((0,),-8,.012,2),((1,),-6,.025,2),((0,1),-7,.02,2),((0,2),-7,.02,2),((0,3),-7,.02,2),((0,1,2,3),-8,.012,2),((0,1,2,3),-11,.006,2),((0,1,2,3),-7,.07,2),((1,2,3),-8,.016,2),((0,1,2,3),-10,.025,1)]
scores=[{},{}]
for si,val in enumerate((np.arange(52,72),np.arange(0,72,4),np.arange(2,72,4))):
 tr=np.setdiff1d(np.arange(72),val);n=len(tr); u=dict(study,train={k:t[k][tr] for k in ('x','y','traj')},test={'x':t['x'][val]}); K=mod['_all_kernels'](u)
 for head,key in enumerate(('y','traj')):
  y=t[key][tr]; actual=t[key][val]; q=np.linalg.svd(y,full_matrices=False)[2][:2] if head else None
  matrices=[mod['_covariance'](np.concatenate([K[j] for j in gs]),y,n,prior,st,mode if head else 0) for gs,prior,st,mode in params];
  lists=[(i,) for i in range(len(params))]+[(0,i) for i in range(1,len(params))]+[(0,4,7,8),(0,3,4,8,10),(0,1,4,6,7),(0,4,5,6,7,8,9,10,11),tuple(range(len(params)))];
  for ids,noise_mult in itertools.product(lists,(0.,1.,5.)):
   M=np.mean([matrices[i][0] for i in ids],axis=0);noise=np.mean([matrices[i][1] for i in ids]); F=mod['_factor'](M,n,noise_mult*noise);p=F.shape[1]
   if head:
    blocks=[np.r_[np.ones(4),np.full(p-8,j),np.full(4,j*k)] for j in (3.,30.,300.) for k in (.3,3.,30.)]; pred=F[n:]@fit(F[:n],y@q.T,blocks)@q
   else:
    blocks=[np.r_[np.ones(4),np.full(10,k),np.full(p-18,j),np.full(4,j*3)] for k in (.15,1.,10.) for j in (3.,30.,300.)];pred=F[n:]@fit(F[:n],y,blocks)
   scores[head].setdefault((ids,noise_mult),[]).append(float(np.mean(abs(pred-actual))))
 print('completed stress slice',si,flush=True)
chosen=[]
for head in range(2):
 result=sorted(scores[head].items(),key=lambda kv:(max(0.,kv[1][0]-.0378)+.22*max(0.,max(kv[1])-.055),np.mean(kv[1])))
 print('head',head,'best',result[:5],flush=True); chosen.append(result[0])
old=np.array([.062890,.052382]); new=np.array([v[0] for k,v in chosen]); print('official comparisons',old,new,flush=True)
source=Path('/tmp/v03_14.py').read_text(); first=source.index('def _affine'); second=source.index('def _latent')
source=source[:first]+'''def _affine(x):
    return np.column_stack((np.ones(len(x)),x))


'''+source[second:source.index('def _profile_features')]+Path('/tmp/add14.txt').read_text()
source+='\n\n_PROFILES = '+repr(params)+'\n_SELECTION = '+repr([key for key,vals in chosen])+'''\n\n\ndef run_node(inputs: dict) -> dict:
    study=inputs['__input__']
    n=len(study['train']['x'])
    kernels=_all_kernels(study)
    features=[]
    for head,key in enumerate(('y','traj')):
        ids,mult=_SELECTION[head]
        models=[]
        for i in ids:
            gs,prior,strength,mode=_PROFILES[i]
            K=np.concatenate([kernels[j] for j in gs])
            models.append(_covariance(K,study['train'][key],n,prior,strength,mode if head else 0))
        M=np.mean([v[0] for v in models],axis=0)
        noise=np.mean([v[1] for v in models])
        features.append(_factor(M,n,mult*noise))
    r,z=features
    return {'readout_train':r[:n], 'readout_test':r[n:], 'z':z[:n], 'zt':z[n:]}
'''
if np.max(new)<=.038 or np.max(new)<np.max(old):
 (P/'node_00.py').write_text(source)
