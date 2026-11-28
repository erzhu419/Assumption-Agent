import sys,pickle,importlib,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
f=[importlib.import_module(f'node_{i:02d}').run_node for i in range(3)]
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); rows=np.arange(n)
weights=np.array([-.5,0,.15,.3,.45,.55,.65,.8,1,1.25,1.5]); records=[]
splits=[('gate',rows[:n-20],rows[n-20:])]+[(f'block{i}',np.setdiff1d(rows,v),v) for i,v in enumerate(np.array_split(rows,6))]
for seed in range(4):
 for i,v in enumerate(np.array_split(np.random.default_rng(seed).permutation(n),4)):
  splits.append((f'random{seed}.{i}',np.setdiff1d(rows,v),v))
for name,tr,va in splits:
 sh=f[0]({'__input__':{**s,'train':{k:t[k][tr] for k in ('x','y','traj')},'test':{'x':t['x'][va]}}}); p=f[1]({'node_00':sh})['poly']; res=f[2]({'node_00':sh}); q=res['smooth']; st=(.4*p[:,1:]+.6*q[:,1:])@res['vt']; y=t['y'][va]; a=np.abs(q[:,0,None]+(p[:,0]-q[:,0])[:,None]*weights-y[:,None]); b=np.abs(st-t['traj'][va]); records.append((name,va,p[:,0],q[:,0],y,a,b)); print(name,'n',len(va),'scalar',np.round(a.mean(0),4),'traj',round(b.mean(),5),flush=True)
print('weights',weights); print('aggr',np.concatenate([r[-2] for r in records[1:]]).mean(0)); print('worst',np.max([r[-2].mean(0) for r in records],axis=0)); print('telemetry',s['telemetry_vec']); pickle.dump(records,open('/tmp/blend_m102_records.pkl','wb'))
