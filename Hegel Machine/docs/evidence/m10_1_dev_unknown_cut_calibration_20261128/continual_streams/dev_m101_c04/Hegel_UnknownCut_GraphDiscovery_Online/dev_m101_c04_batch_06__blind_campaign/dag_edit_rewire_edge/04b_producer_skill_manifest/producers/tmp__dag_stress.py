from dag_probe import *
import copy
sink = runpy.run_path(p+'node_01.py')['run_node']
raw=s['train']['x']; rows=np.arange(len(raw)); size=20
blocks=[('official',rows[-size:]),('first',rows[:size]),('middle',rows[20:40]),('late',rows[40:60]),('high_radius',np.argsort(np.linalg.norm(u, axis=1))[-size:]),('high_mixed_projection',np.argsort(np.abs(z[:,0]))[-size:])]
for name, v in blocks:
    t=np.setdiff1d(rows,v)
    study=copy.deepcopy(s)
    study['train']={key:s['train'][key][t] for key in ['x','y','traj']}
    study['train']['summary']=np.array([Y[t].mean(),Y[t].std(),R[t,-1].mean(),R[t,-1].std()])
    study['test']={'x':raw[v]}
    pre=a['run_node']({'__input__':study}); mixed=b['run_node']({'node_02':pre})
    out=sink({'__input__':study,'node_02':pre,'node_00':mixed})
    y=out['y_hat']; traj=out['state_pred']; last=traj[:,-1]
    np.testing.assert_allclose(out['diagnostics_vec'],[y.mean(),y.std(),last.mean(),last.std()])
    print(name, 'y MAE',round(np.abs(y-Y[v]).mean(),6), 'traj MAE',round(np.abs(traj-R[v]).mean(),6), 'worst horizon',round(np.abs(traj-R[v]).mean(axis=0).max(),6))
