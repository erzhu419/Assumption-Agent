exec(open('/tmp/stress.py').read().split('for name,va in folds:')[0].replace("'/holdout_task/batch_solver.py'","'/tmp/sparse_baseline.py'"))
configs=[(b,r,'sat') for b in [.22,.5,.75,1,1.25] for r in [.015,.08,.3,1.]]+[(b,r,'tri') for b in [-1,-.5,0,.5] for r in [.08,.3,1.]]
err=[]
for name,va in folds:
 ix=np.setdiff1d(allidx,va); r=dict(s,train={k:v[ix] if k in ('x','y','traj') else v for k,v in tr.items()},test={'x':tr['x'][va]}); p=m.solve_batch(r); target=np.column_stack([tr['y'][va],tr['traj'][va]]); base=np.column_stack([p['y_hat'],p['state_pred']]); err.append((base-target,np.array([m._panel_pred(r,*c)-target for c in configs])))
for j in [0,1]:
 rows=[]
 for i,c in enumerate(configs):
  for w in [.25,.4,.5,.65,.8,1.]:
   e=np.array([np.abs(((1-w)*a+w*b[i])[:, :(1) if j==0 else None] if j==0 else ((1-w)*a+w*b[i])[:,1:]).mean() for a,b in err]);
   if e[0]<.0378:rows.append((e[1:5].mean()+.55*e[5:].max()+.15*e[0],w,c,e.round(5)))
 print(j,*sorted(rows,key=lambda x:x[0])[:9],sep='\n')
