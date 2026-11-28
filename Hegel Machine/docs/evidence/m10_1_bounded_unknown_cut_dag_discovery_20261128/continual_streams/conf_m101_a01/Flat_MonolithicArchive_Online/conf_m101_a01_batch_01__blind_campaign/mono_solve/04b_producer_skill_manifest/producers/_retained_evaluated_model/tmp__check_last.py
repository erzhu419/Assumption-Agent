import pickle, pathlib, runpy
import numpy as np
p=pathlib.Path('/holdout_task/batch_solver.py'); base=p.read_text(); addition=pathlib.Path('/tmp/projected.py').read_text(); scope={}; exec(base+'\n'+addition,scope)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x,y=t['x'],t['y']; ix=np.arange(len(y))
splits=[(ix[:52],ix[52:]),(ix[:56],ix[56:])]
for seed in range(22,32):
    q=np.random.default_rng(seed).permutation(ix); splits.append((q[:52],q[52:]))
errs=[]
for i,j in splits:
    b=scope['_smooth'](x[i],x[j],y[i],1,'iso'); models=[scope['_projected_readout'](x[i],x[j],y[i],g) for g in range(3)]; c=np.mean(models,axis=0)
    errs.append([np.mean(abs(v-y[j])) for v in (b,*models,c,(b+c)/2)])
e=np.asarray(errs); print('scalar MAE columns: baseline, projected x3, projected pool, mixed pool'); print(np.vstack((e[:2],e[2:].mean(0),e[2:].max(0))))
ks=[k for k in range(1,6) if e[0,k]<e[0,0] and (e[2:,k].mean()<e[2:,0].mean() or e[0,k]<=.038) and e[2:,k].max()<=e[2:,0].max()+.005]; k=min(ks,key=lambda k:(e[0,k]>.038,e[2:,k].mean())) if ks else 0
if k:
    a=base.index('    y = np.mean('); b=base.index('    amp =',a)
    expr=f"_projected_readout(x, z, train['y'], {k-1})" if k<=3 else "np.mean([_projected_readout(x, z, train['y'], k) for k in range(3)], axis=0)"
    if k==5: expr=f"0.5 * (_smooth(x, z, train['y'], 1, 'iso') + {expr})"
    p.write_text(base[:a]+'    y = '+expr+'\n'+base[b:]+'\n'+addition)
print('Selected column',k)
