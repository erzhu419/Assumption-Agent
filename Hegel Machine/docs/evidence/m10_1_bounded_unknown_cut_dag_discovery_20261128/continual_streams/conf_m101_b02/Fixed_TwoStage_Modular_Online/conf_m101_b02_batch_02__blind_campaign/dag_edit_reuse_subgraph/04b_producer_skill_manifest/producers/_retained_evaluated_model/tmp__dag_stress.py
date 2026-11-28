import pickle, runpy
import numpy as np
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
m=runpy.run_path(p+'node_00.py'); u=m['run_node']; v=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']
t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; n=len(x)
c=s['panel_c']; a=m['_cross_terms'](c['u']); b=c['v1']; r=np.array([b[i]-a[i]@np.linalg.lstsq(np.delete(a,i,axis=0),np.delete(b,i,axis=0),rcond=None)[0] for i in range(len(a))]); print('C v1 sparse leave-one-out MAE by channel',np.abs(r).mean(0))
rng=np.random.default_rng(101)
orders={'public':np.arange(n)}
orders.update({f'random_{i}':rng.permutation(n) for i in range(80)})
orders.update({f'block_{i}':np.roll(np.arange(n),i) for i in range(0,n,4)})
for j in range(4):
    ix=np.argsort(x[:,j]); orders[f'tail_high_{j}']=ix; orders[f'tail_low_{j}']=ix[::-1]
orders['tail_radius']=np.argsort(np.linalg.norm(x-np.median(x,axis=0),axis=1))
results={}
for key,ix in orders.items():
    a,b=ix[:-20],ix[-20:]
    f=u({'__input__':{**s,'train':{'x':x[a],'y':y[a],'traj':tr[a]},'test':{'x':x[b]}}})
    g=f['g']
    for name in ('artifact','2x','4x','8x','unit','whiten1','whiten2'):
        if name=='artifact': h=g
        elif name.endswith('x'): h=np.column_stack((g[:,0],float(name[:-1])*g[:,1:]))
        else:
            center=g[:len(a),1:].mean(0); z=g[:,1:]-center; scale=g[:len(a),1:].std(0); z=z/scale
            if name.startswith('whiten'):
                _,sigma,Vt=np.linalg.svd(z[:len(a)],full_matrices=False); z=(z@Vt.T)/np.maximum(sigma/np.sqrt(len(a)), float(name[-1])/10)
            h=np.column_stack((g[:,0],z))
        o=v({'node_00':{**f,'g':h}}); results[name,key]=(abs(o['y_hat']-y[b]).mean(),abs(o['state_pred']-tr[b]).mean())
for name in ('artifact','2x','4x','8x','unit','whiten1','whiten2'):
    print(name,[(group,[(k,round(results[name,k][j],6)) for j in range(2) for k in [max((k for k in orders if k.startswith(group)),key=lambda k:results[name,k][j])]]) for group in ('public','random','block','tail')])
