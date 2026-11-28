import numpy as np,pickle,glob,runpy
from itertools import combinations
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; z=tr['traj']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*batch_08_solver.py')[0]); n=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); u=n['run_node']({'__input__':s}); coef=a['_fit_calibration'](s['panel_a']); print('calibration',[(t,j,coef[i,j]) for i,t in enumerate(a['_TERMS']) for j in range(4) if coef[i,j]!=0]); print('p',u['p'],'panel-B MAE',abs(n['_b_features'](s['panel_b']['u'],u['p'])@u['w']-s['panel_b']['v']).mean())
transform=lambda q:n['_b_features'](q,u['p'])@u['w']
q=a['_invert'](tr['x'],coef); b=transform(q); c=b*np.roll(q,-1,axis=1); h=s['panel_c']; hb=transform(h['u']); print('panel C',abs(hb-h['v1']).mean(),[(j,abs(h['v1']*np.roll(h['u'],j,axis=1)-h['v2']).mean()) for j in range(4)])
o=np.ones(len(q)); aff=np.c_[o,b,c]; state=np.c_[o,b,c.sum(1)]; folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(42).permutation(72),5));
def score(f,target=y):
 res=[]
 for te in folds:
  it=np.setdiff1d(np.arange(len(q)),te); res.append(abs(f[te]@ls(f[it],target[it])-target[te]).mean())
 return (res[0],max(res[1:]),np.mean(res[1:]),abs(f@ls(f,target)-target).mean())
print('state',score(state,z)); print('affine',score(aff)); quad=np.c_[o,q,*[q[:,i]*q[:,j] for i in range(4) for j in range(i,4)]]; print('quad',score(quad));
tel=s['telemetry_vec']; gates=[]
for j in range(4):
 h=q[:,j]; gates += [(f'rat{j}',1/(1+tel[j]*h*h)),(f'rat{j}a',1/(1+1.25*h*h))]; gates += [(f'tanh{j}:{k}',np.tanh(k*h)) for k in [2,4,8,12,20]]
print('archive08',score(np.c_[o,b*gates[14][1][:,None],c,b[:,3]*q[:,2]**3*gates[14][1]]))
res=[]
for nm,g in gates:
 for desc,f in [('gB',np.c_[o,b*g[:,None],c]),('aff+gB',np.c_[aff,b*g[:,None]]),('aff+gA',np.c_[aff,np.c_[o,b]*g[:,None]]),('aff+gAll',np.c_[aff,aff*g[:,None]]),('quad+gq',np.c_[quad,quad[:,:5]*g[:,None]])]:
  res.append((score(f),nm,desc))
 for nm2,g2 in gates:
  if nm2>nm: res.append((score(np.c_[aff,b*g[:,None],b*g2[:,None]]),nm,nm2))
for k in [0,1,3]:
 print('sort',k);print(*sorted(res,key=lambda r:r[0][k])[:12],sep='\n')
