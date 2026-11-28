exec(open('/tmp/test_solver.py').read().split("print('endpoint coefficients'")[0])
ix=np.arange(len(y)); o,g=m._calibration(s['panel_a']); z=(r['x']-o)/g
splits=[('ordered',ix)]+[(f'random{k}',np.random.default_rng(k).permutation(len(y))) for k in range(10)]+[('high amplitude',np.argsort(np.max(abs(z),axis=1)))]+[(f'axis{k} sign tail {q}',np.argsort(q*z[:,k])) for k in range(4) for q in (-1,1)]
vals=[]
for name,p in splits:
 i,j=p[:56],p[56:]; result=m.solve_batch(dict(s,train={k:r[k][i] for k in ['x','y','traj']},test={'x':r['x'][j]})); ey=abs(result['y_hat']-y[j]); es=abs(result['state_pred']-Y[j]); scores=[ey.mean(),es.mean(),es.mean(0).max(),es[:,-1].mean()]; vals.append(scores)
 if name in ('ordered','high amplitude'): print(name,np.round(scores,7))
 if name=='ordered':
  additive=aff(np.c_[A,B]); ys=additive[j]@m._fit(additive[i],y[i]); ss,ff=B,A; ba,(cs,cf)=m._fit_modes(ss[i],ff[i],T,Y[i]); st=np.c_[ss[j]@cs,ff[j]@cf]@ba
  print('interaction ablation ordered',np.mean(abs(ys-y[j])),np.mean(abs(st-Y[j])))
v=np.array(vals); print('worst original slices',v[:12].max(0)); print('worst all slices',v.max(0),[splits[k][0] for k in v.argmax(0)])
assert np.all(v.max(0)<0.009)
