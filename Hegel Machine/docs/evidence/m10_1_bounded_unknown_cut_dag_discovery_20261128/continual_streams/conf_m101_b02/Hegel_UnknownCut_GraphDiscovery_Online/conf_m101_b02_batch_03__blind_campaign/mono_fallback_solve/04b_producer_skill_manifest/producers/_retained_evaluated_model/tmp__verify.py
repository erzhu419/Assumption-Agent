import importlib.util,pickle,numpy as np
from sklearn.model_selection import KFold
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x)
folds=[('52/20',np.arange(52),np.arange(52,n)),('56/16',np.arange(56),np.arange(56,n))]+[(f'{seed}:{k}',i,j) for seed in (123,319,711) for k,(i,j) in enumerate(KFold(4,shuffle=True,random_state=seed).split(x))]
for name,i,j in folds:
 d=dict(s,train=dict(tr,**{k:tr[k][i] for k in ['x','y','traj']}),test={'x':x[j]}); o=m.solve_batch(d); r=abs(o['state_pred']-tr['traj'][j]); print(name,round(abs(o['y_hat']-tr['y'][j]).mean(),5),round(r.mean(),5),round(r.mean(0).max(),5))
g,e=m._features(s,x); h=m._time_basis(s['eval_times'],tr['traj']); w=np.c_[g,e]; lab=np.c_[tr['y'],np.linalg.lstsq(h,tr['traj'].T,rcond=None)[0].T]; print('basis',h[-1]);
from sklearn.linear_model import ARDRegression
ij=list(zip(*np.triu_indices(12))); names=[f'{p}{j}' for p in 'fcd' for j in range(4)]; names += [names[i]+'*'+names[j] for i,j in ij]; names += [f'{fun}{p}{j}' for p in 'fcd' for fun in ('sin','cos') for j in range(4)]
for a,b in ((g,lab[:,0]),(w,lab[:,1]),(w,lab[:,2])):
 mdl=ARDRegression(threshold_lambda=10000).fit((a-a.mean(0))/a.std(0),b); c=mdl.coef_/a.std(0); print('active',[(names[i],round(c[i],5)) for i in np.argsort(abs(mdl.coef_))[-15:][::-1] if abs(mdl.coef_[i])>0]); print('fit',np.mean(abs(mdl.predict((a-a.mean(0))/a.std(0))-b)))
