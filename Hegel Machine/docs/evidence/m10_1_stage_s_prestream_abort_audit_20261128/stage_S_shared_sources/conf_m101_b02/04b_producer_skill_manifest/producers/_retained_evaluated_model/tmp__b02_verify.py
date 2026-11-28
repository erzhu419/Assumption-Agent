import sys,pickle,importlib.util,numpy as np
p=sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'
a=importlib.util.spec_from_file_location('candidate',p);m=importlib.util.module_from_spec(a);a.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];x=tr['x'];n=len(x);I0=np.arange(n)
z=(x-x.mean(0))/x.std(0);rad=np.sum(z*z,axis=1)
folds=[('tail',I0[56:]),*[(str(i),j) for i,j in enumerate(np.array_split(np.random.default_rng(34).permutation(n),4))],('edge',np.argsort(rad)[-12:])]
for name,J in folds:
 I=np.setdiff1d(I0,J); sub={**s,'train':{k:v[I] for k,v in tr.items() if k!='summary'},'test':{'x':x[J]}}
 sub['train']['summary']=np.array([tr['y'][I].mean(),tr['y'][I].std(),tr['traj'][I,-1].mean(),tr['traj'][I,-1].std()]); y,T=m._predict(sub,report=(name=='tail'));err=abs(T-tr['traj'][J]);print(name, 'y, avgT, finalT, worst horizon:',np.round([abs(y-tr['y'][J]).mean(),err.mean(),err[:,-1].mean(),err.mean(0).max()],6))
