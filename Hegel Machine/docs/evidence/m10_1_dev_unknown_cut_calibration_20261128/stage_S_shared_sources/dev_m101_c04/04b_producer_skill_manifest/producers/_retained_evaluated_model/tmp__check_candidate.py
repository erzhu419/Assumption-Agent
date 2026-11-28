import importlib.util, pickle, numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
x,y,T=(s['train'][k] for k in ('x','y','traj')); z=m._coordinates(x,m._calibration(s)); t=s['eval_times']; test=m._coordinates(s['test']['x'],m._calibration(s)); n=len(x)
splits=[('56/16',np.arange(56,72))]
splits += [(f'block{i}',np.arange(i,i+12)) for i in range(0,72,12)]
for i in range(4):
 j=np.argsort(z[:,i]); splits += [(f'min{i}',j[:12]),(f'max{i}',j[-12:])]
splits += [(f'random{i}',j) for i,j in enumerate(np.array_split(np.random.default_rng(988).permutation(n),6))]
D=m._affine(z); C=m._cross(z); S=np.c_[D,C]; norms=np.linalg.norm((z-z.mean(0))/z.std(0),axis=1)
splits += [('radial',np.argsort(norms)[-12:]),('cross_min',np.argsort(C)[:12]),('cross_max',np.argsort(C)[-12:])]
ys={k:[] for k in ['shared','no_intercept','fixed18','fixed18_no_intercept']}; ts={k:[] for k in ['exp2','exp2_no_intercept','affine']}
for name,j in splits:
 i=np.setdiff1d(np.arange(n),j)
 out=m.solve_batch({**s,'train':{'x':x[i],'y':y[i],'traj':T[i]},'test':{'x':x[j]}})
 yt=out['y_hat']; st=out['state_pred']; ys['shared'].append(np.mean(abs(yt-y[j]))); ts['exp2'].append(np.mean(abs(st-T[j]),axis=0))
 for kind,F,b in [('no_intercept',S[:,1:],0*C),('fixed18',D,.18*C),('fixed18_no_intercept',z,.18*C)]:
  ys[kind].append(np.mean(abs(F[j]@m._ls(F[i],(y-b)[i])+b[j]-y[j])))
 ts['exp2_no_intercept'].append(np.mean(abs(z[j]@m._relaxation(z[i],T[i],t)-T[j]),axis=0))
 ts['affine'].append(np.mean(abs(D[j]@m._ls(D[i],T[i])-T[j]),axis=0))
for kind,arr in {**ys,**ts}.items():
 a=np.asarray(arr); j=np.unravel_index(a.argmax(),a.shape); print(kind,'56/16',a[0].mean(),a[0].max(),'all mean',a.mean(),'worst',a[j],splits[j[0]][0],j[1:],'worst range slices',a[7:15].max(),'worst composed slice',a[-3:].max())
print('test coordinate ranges',test.min(0),test.max(0),'train',z.min(0),z.max(0)); print('test cross range',m._cross(test).min(),m._cross(test).max(),'train',C.min(),C.max())
