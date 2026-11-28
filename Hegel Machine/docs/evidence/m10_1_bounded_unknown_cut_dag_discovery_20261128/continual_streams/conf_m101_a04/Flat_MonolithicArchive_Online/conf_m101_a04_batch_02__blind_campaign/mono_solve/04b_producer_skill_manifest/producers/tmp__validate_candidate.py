import importlib.util, pickle, sys
import numpy as np
spec = importlib.util.spec_from_file_location('candidate',sys.argv[1]); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; n=len(x)
scalar,g,h=mod._design(s); scalar,g,h=scalar[:n],g[:n],h[:n]; full=np.c_[g,h]; clocks=s['eval_times']
print('clocks',clocks)
coef=mod._fit(full,t); print('unconstrained first branch profiles, normalized by final:'); print(np.round(coef[:5]/coef[:5,-1:],4)); print('second:'); print(np.round(coef[5:]/coef[5:,-1:],4))
rng=np.random.default_rng(6041); radius=np.linalg.norm((x-x.mean(0))/x.std(0),axis=1)
folds=[('public52',np.arange(52,72)),('tail56',np.arange(56,72)),('radius',np.argsort(radius)[-16:])]
for i in range(x.shape[1]):
 for sign in (-1,1): folds.append((f'axis{i}/{sign}',np.argsort(sign*x[:,i])[-16:]))
folds += [(f'random{i}',e) for i in range(10) for e in [rng.permutation(n)[:16]]]
for label,e in folds:
 a=np.setdiff1d(np.arange(n),e); v={**s,'train':{**tr,'x':x[a],'y':y[a],'traj':t[a]},'test':{'x':x[e]}}
 out=mod.solve_batch(v); d=abs(out['state_pred']-t[e]); p=full[e]@mod._fit(full[a],t[a]); du=abs(p-t[e]); print(label,*(f'{z:.6f}' for z in (abs(out['y_hat']-y[e]).mean(),d.mean(),d.mean(0).max(),du.mean(),du.mean(0).max())))
