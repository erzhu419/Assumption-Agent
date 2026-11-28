import itertools as it, pickle, runpy
import numpy as np
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
m=runpy.run_path(p+'node_00.py'); sink=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']
t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; n=len(x)
f=m['run_node']({'__input__':{**s,'test':{'x':x[:0]}}}); g=f['g']
c=m['_calibration'](s['panel_a']); z=np.linalg.solve(c[1:].T,(x-c[0]).T).T
cub=list(it.combinations_with_replacement(range(4),3))
pure=np.array([15+i for i,k in enumerate(cub) if len(set(k))==1]); mixed=np.array([15+i for i,k in enumerate(cub) if len(set(k))>1]); quartic=np.column_stack([np.prod(z[:,k],axis=1) for k in it.combinations_with_replacement(range(4),4)])
rng=np.random.default_rng(7009)
orders={'public':np.arange(n), **{f'random_{i}':rng.permutation(n) for i in range(160)}, **{f'block_{i}':np.roll(np.arange(n),i) for i in range(0,n,4)}}
for j in range(4):
    ix=np.argsort(x[:,j]); orders[f'tail_hi{j}']=ix; orders[f'tail_lo{j}']=ix[::-1]
orders['tail_radius']=np.argsort(np.linalg.norm(x-np.median(x,axis=0),axis=1))
models={'baseline':g}
for a in (.03,.06,.12,.25,.5):
    h=g.copy(); h[:,pure]*=a; models[f'pure_{a}']=h
for a in (.1,.25,.5,2.0):
    h=g.copy(); h[:,mixed]*=a; models[f'mixed_{a}']=h
for a in (.002,.005,.01,.02,.05): models[f'quartic_{a}']=np.column_stack([g,a*quartic])
for name,h in models.items():
    e={}
    for key,ix in orders.items():
        a,b=ix[:-20],ix[-20:]; o=sink({'node_00':{'g':h[ix],'fd':f['fd'][ix],'cd':f['cd'][ix],'y':y[a],'traj':tr[a]}})
        e[key]=np.array([abs(o['y_hat']-y[b]).mean(),abs(o['state_pred']-tr[b]).mean()])
    print(name, 'public', e['public'].round(6), *[(group,np.max([v for k,v in e.items() if k.startswith(group)],axis=0).round(6)) for group in ('random','block','tail')], 'tail4low',e['tail_lo3'].round(6))
