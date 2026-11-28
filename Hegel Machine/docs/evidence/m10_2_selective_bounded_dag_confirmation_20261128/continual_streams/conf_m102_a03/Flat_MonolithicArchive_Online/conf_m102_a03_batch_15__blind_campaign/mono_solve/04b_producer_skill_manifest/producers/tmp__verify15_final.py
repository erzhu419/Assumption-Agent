import pickle
import runpy
import numpy as np

p='/holdout_task/'
m=runpy.run_path(p+'batch_solver.py')
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
tr=s['train']; x=tr['x']; ids=np.arange(len(x)); t=s['eval_times']; rng=np.random.default_rng(782)
z=m['_decode'](x,m['_calibration'](s['panel_a'])); b=m['_operator'](s['panel_b'],z)
splits=[('fixed',ids[52:])]+[(f'random_{j}',rng.permutation(ids)[52:]) for j in range(60)]
for name,values in [('x',x),('z',z),('b',b)]:
    for j,channel in enumerate(values.T):
        i=np.argsort(channel)
        splits.extend([(f'{name}{j}_low',i[:20]),(f'{name}{j}_high',i[-20:])])
    i=np.argsort(np.linalg.norm(values-np.median(values,axis=0),axis=1))
    splits.append((name+'_radial',i[-20:]))
metrics=[]
for name,v in splits:
    a=np.setdiff1d(ids,v); y=tr['y'][a]; T=tr['traj'][a]; tail=T[:,-1]
    train=dict(x=x[a],y=y,traj=T,summary=np.array([y.mean(),y.std(),tail.mean(),tail.std()]))
    out=m['solve_batch'](dict(s,train=train,test={'x':x[v]}))
    yp=out['y_hat']; P=out['state_pred']; tail=P[:,-1]
    np.testing.assert_allclose(out['diagnostics_vec'],[yp.mean(),yp.std(),tail.mean(),tail.std()],rtol=0,atol=1e-14)
    d=abs(yp-tr['y'][v]); e=abs(P-tr['traj'][v])
    metrics.append([d.mean(),e.mean(),e.mean(0).max()])
M=np.array(metrics)
print('Columns: scalar MAE, trajectory MAE, worst-timestamp MAE')
print('Prescribed 52/20:',M[0]); print('Worst across',len(M),'splits:',M.max(0)); print('Worst slice names:',[splits[j][0] for j in M.argmax(0)])
assert np.isfinite(M).all() and M.max()<0.038
