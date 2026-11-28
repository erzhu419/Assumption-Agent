import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; x=T['x']; y=T['y']; h=T['traj']; t=s['eval_times']; n=len(x)
z,c,d=m._features(s,x); dot=(c*d).sum(1,keepdims=True); p3=m._powers(z,3); cross=(c[:,:,None]*d[:,None,:]).reshape(n,-1)
p,r,w,e=m._temporal(t,h,[np.c_[c,dot],p3,np.c_[d,c,cross]],inspect=True); print('temporal',r,e,'fast and cubic',w[:25], 'slow cross',w[-16:].reshape(4,4))
folds=[('56/16',np.arange(56,72))]+[(f'fold {k}',ix) for k,ix in enumerate(np.array_split(np.random.default_rng(123).permutation(n),5))]+[('radial tail',np.argsort(np.linalg.norm(z,axis=1))[-14:])];
for label,te in folds:
 tr=np.setdiff1d(np.arange(n),te); sub={**s,'train':{**T,**{k:T[k][tr] for k in ('x','y','traj')}},'test':{'x':x[te]}}; q=m.solve_batch(sub); err=abs(q['state_pred']-h[te]); print(label,np.mean(abs(q['y_hat']-y[te])),err.mean(),err[:,-1].mean(),np.round(err.mean(0),4));
