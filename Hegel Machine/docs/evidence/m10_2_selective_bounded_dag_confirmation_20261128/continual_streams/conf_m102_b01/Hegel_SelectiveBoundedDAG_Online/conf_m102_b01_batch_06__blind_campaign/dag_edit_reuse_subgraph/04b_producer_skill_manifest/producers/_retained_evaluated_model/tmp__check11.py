exec(open('/tmp/fit07.py').read().split('for name,b')[0]); fn=runpy.run_path('/holdout_task/node_01.py')['_fit'];
for a,v in folds:
 yp,tp=fn(w[np.r_[a,v]],y[a],tr[a],p,True); print('outer MAE',abs(yp-y[v]).mean(),abs(tp-tr[v]).mean(), 'horizon MAE',abs(tp-tr[v]).mean(0),flush=True)
