import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['x']); ps=[]; truths=[]; times=[]; traces=[]
for ix in [np.arange(n)]+[np.random.default_rng(i+177).permutation(n) for i in range(16)]:
 a,b=ix[:52],ix[52:]; ss={**s,'train':{**t,**{k:t[k][a] for k in ('x','y','traj')}},'test':{'x':t['x'][b]}}; yp,tp,old=m._components(ss); ps.append(np.r_[yp,old['y_hat'][None,:]]); truths.append(t['y'][b]); times.append([tp,old['state_pred']]); traces.append(t['traj'][b]);
p=np.array(ps); y=np.array(truths); T=np.array(times); t=np.array(traces)
print('scalar components 52/20',np.mean(abs(p-y[:,None,:]),axis=-1)[0]);
for w in (0,.1,.2,.3,.5,1):
 pred=(1-w)*p[:,:4].mean(1)+w*p[:,4:7].mean(1); e=np.mean(abs(pred-y),axis=1); print('scalar blend',w,e[0],e.mean(),e.max());
for w in (0,.2,.5,1):
 e=abs((1-w)*T[:,0]+w*T[:,1]-t); a=e.mean((1,2)); print('trajectory blend',w,a[0],a.mean(),a.max(),'worst horizon',e.mean(1).max());
np.savez('/tmp/gate_preds.npz',p=p,y=y,T=T,t=t)
