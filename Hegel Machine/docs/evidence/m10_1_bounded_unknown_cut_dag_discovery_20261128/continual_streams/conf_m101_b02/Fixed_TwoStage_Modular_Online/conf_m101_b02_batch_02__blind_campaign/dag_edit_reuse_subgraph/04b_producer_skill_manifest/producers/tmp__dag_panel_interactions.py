import pickle, runpy
import numpy as np
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
u=runpy.run_path(p+'node_00.py')['run_node']; sink=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']
t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; n=len(x)
f=u({'__input__':{**s,'test':{'x':x[:0]}}}); g,fd,cd=[f[k] for k in ('g','fd','cd')]; v,w=np.hsplit(cd,2)
paired=np.column_stack((fd*v,fd*w)); all_products=(fd[:,:,None]*cd[:,None,:]).reshape(n,-1)
models={'baseline':g}
for name,z in [('paired',paired),('all',all_products)]:
    for a in (0.1,0.25,0.5,1.0): models[f'{name}_{a}']=np.column_stack((g[:,:15],a*g[:,15:],z))
rng=np.random.default_rng(83076)
orders={'public':np.arange(n),**{f'random_{i}':rng.permutation(n) for i in range(100)},**{f'block_{i}':np.roll(np.arange(n),i) for i in range(0,n,4)}}
for j in range(4):
    k=np.argsort(x[:,j]); orders[f'tail_hi{j}']=k; orders[f'tail_lo{j}']=k[::-1]
orders['tail_radius']=np.argsort(np.linalg.norm(x-np.median(x,axis=0),axis=1))
for name,h in models.items():
    e={}
    for key,ix in orders.items():
        a,b=ix[:-20],ix[-20:]; o=sink({'node_00':{'g':h[ix],'fd':fd[ix],'cd':cd[ix],'y':y[a],'traj':tr[a]}})
        e[key]=[abs(o['y_hat']-y[b]).mean(),abs(o['state_pred']-tr[b]).mean()]
    print(name,np.round(e['public'],6),*[np.max([v for k,v in e.items() if k.startswith(prefix)],axis=0).round(6) for prefix in ('random','block','tail')])
