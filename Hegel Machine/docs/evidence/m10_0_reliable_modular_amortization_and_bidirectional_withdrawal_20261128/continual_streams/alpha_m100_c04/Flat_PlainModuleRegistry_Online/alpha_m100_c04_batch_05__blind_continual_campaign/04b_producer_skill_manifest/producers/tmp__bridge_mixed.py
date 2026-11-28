import pickle,numpy as np,itertools
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; g=np.max(abs(y),axis=1)<5; x,y=x[g],y[g]; axis=(x!=0).sum(1)==1; pairs=list(itertools.combinations(range(4),2)); n=len(x)
for j in range(4):
 print('\nchannel',j)
 for name, fs in [('linear',[]),('quadratic',[x*x]),('sin_cos',[np.sin(2.5*x),1-np.cos(2.5*x)]),('pair_quadratic',[x*x,np.array([x[:,a]*x[:,b] for a,b in pairs]).T]),('cubic',[x*x,np.array([np.prod(x[:,p],axis=1) for p in itertools.combinations_with_replacement(range(4),3)]).T,np.array([x[:,a]*x[:,b] for a,b in pairs]).T]),('pair_sines',[np.sin(2.5*x),1-np.cos(2.5*x),*[np.array([f(2.5*(x[:,a]+sg*x[:,b])) for a,b in pairs]).T for f in [np.sin,lambda t:1-np.cos(t)] for sg in [1,-1]]])]:
  a=np.column_stack([np.ones(n),x,np.sin(2.5*x[:,j]),*fs]); b=np.linalg.lstsq(a,y[:,j],rcond=None)[0]; e=y[:,j]-a@b; loo=np.array([y[k,j]-a[k]@np.linalg.lstsq(np.delete(a,k,0),np.delete(y[:,j],k),rcond=None)[0] for k in range(n)]); print(name,'fit/LOO RMSE',np.round([np.mean(e**2)**.5,np.mean(loo**2)**.5],7),'LOO axis/mixed max',np.round([max(abs(loo[axis])),max(abs(loo[~axis]))],7));
  if name in ('linear','pair_quadratic','pair_sines'): print('coef',np.round(b,4));
 print('ridge profile: compare simple pair sums and differences across the entire mixed design')
 for k in range(4):
  if k==j: continue
  m=axis&(x[:,k]!=0); print(k,'axis differential',np.round(np.c_[x[m,k],y[m,j]],6).tolist())
