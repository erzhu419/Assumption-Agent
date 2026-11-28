import pickle,runpy,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ix=np.arange(n)
up=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; sink=runpy.run_path('/holdout_task/node_01.py')['run_node']
def check(name,folds,panel_drop=False):
    errors=[]
    for i,te in enumerate(folds):
        it=np.setdiff1d(ix,te); ts={**s,'train':{k:tr[k][it] for k in ('x','y','traj')},'test':{'x':tr['x'][te]}}
        if panel_drop:
            ts['panel_a']={k:np.delete(v,i,axis=0) for k,v in s['panel_a'].items()}
        p=sink({'__input__':ts,'node_00':up({'__input__':ts})}); y=p['y_hat']; z=p['state_pred']; assert z.shape==tr['traj'][te].shape and y.shape==tr['y'][te].shape
        np.testing.assert_allclose(p['diagnostics_vec'],[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); errs=np.abs(z-tr['traj'][te]); errors.append([np.abs(y-tr['y'][te]).mean(),errs.mean(),errs.mean(0).max()])
    print(name,'worst scalar, trajectory, horizon',np.max(errors,axis=0)); assert np.max(errors)<.038
check('ordinary random folds',[te for seed in (77,118,231,408,511,1701) for te in np.array_split(np.random.default_rng(seed).permutation(n),4)])
check('ordered folds',np.array_split(ix,4)); tail=np.argsort(np.linalg.norm(tr['x'],axis=1))[-12:]; check('input extremes',[tail]+[np.argsort(tr['x'][:,k])[sl] for k in range(tr['x'].shape[1]) for sl in (slice(0,12),slice(-12,None))]); check('panel jackknife on extreme slice',[tail]*len(s['panel_a']['u']),True)
