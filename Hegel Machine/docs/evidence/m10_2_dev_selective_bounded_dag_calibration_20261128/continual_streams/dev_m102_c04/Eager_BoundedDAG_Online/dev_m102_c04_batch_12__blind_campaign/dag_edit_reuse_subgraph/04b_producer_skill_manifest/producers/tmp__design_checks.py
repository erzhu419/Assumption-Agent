import runpy,pickle,numpy as np
m=runpy.run_path('/holdout_task/node_00.py');sink=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py');temporal=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py');s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x'])-20; a=m['_affine']; ls=m['_ls']; B=ls(a(s['panel_b']['u']),s['panel_b']['v']); q=m['_calibration'](s['panel_a'],tr['x']); print('x and q ranges',tr['x'].min(),tr['x'].max(),q.min(),q.max()); bs=temporal['run_node']({'__input__':dict(s,train={k:(v[:n] if k!='summary' else v) for k,v in tr.items()})});
for name,z in [('q',q),('qB',a(q)@B),('qBr',np.linalg.solve(B[1:].T,(q-B[0]).T).T),('x',tr['x'])]:
 p,r=m['_operators'](s['panel_c'],z);c,o=m['_features'](p,r); tests=[('base',c),('spatial full',np.c_[c,o]),('all pr',np.c_[p,r,(p[:,:,None]*r[:,None,:]).reshape(len(p),16)]),('base+q',np.c_[c,q]),('full+q',np.c_[c,o,q]),('rawpolys',np.c_[z,z*z,m['_cross'](z)])]; print(name)
 for desc,f in tests:
  d=a(f); print(desc,*[round(np.mean(abs(d[n:]@ls(d[:n],y[:n])-y[n:])),6) for y in [tr['y'],tr['traj']]])
 print('sink',np.mean(abs(sink['run_node']({'node_00':{'common':c,'offdiag':o,'y_hat':np.zeros(len(c)-n)},'node_01':bs})['state_pred']-tr['traj'][n:])))
