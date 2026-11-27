exec(open('/tmp/forward98.py').read().split("if __name__")[0])
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];c=d['calibration_load_cells'];x=c['true_channel_loads'];y=c['bridge_raw'];print('axis sweeps:'); print(np.round(np.c_[x,y][:20],4));
a=basis(x); Y=np.c_[y[:,0],y[:,2],y[:,1]+y[:,3],y[:,1]-y[:,3]]
I=[0]+[j+1 for j,q in enumerate(TERMS) if len(q)==1 or len(set(q))==1 and len(q)==3 or q in [(0,1),(1,2),(2,3),(0,3)]]
for k in range(4):
 b=Y[:,k]; A=a[:,I]; rng=np.random.default_rng(98); best=(-1,None)
 for ii in [np.arange(len(x)),np.arange(16)]+[rng.choice(len(x),22,False) for _ in range(600)]:
  q=np.linalg.lstsq(A[ii],b[ii],rcond=None)[0]; r=b-A@q; score=np.sum(abs(r)<.002)
  if score>best[0]:best=(score,q)
 q=best[1]; r=b-A@q; ok=abs(r)<.002
 if sum(ok)>=A.shape[1]:q=np.linalg.lstsq(A[ok],b[ok],rcond=None)[0];r=b-A@q
 print('projection',k,'n clean',best[0], 'q',[(('bias' if j==0 else TERMS[j-1]),round(v,5)) for j,v in zip(I,q) if abs(v)>.001]);print('resids',np.quantile(abs(r),[.5,.75,.85,.9,.95,.99,1]));
 if k==3: print(np.round(np.c_[x,y[:,1],y[:,3],r][abs(r)>.002],3))
print('coupon raw quantiles',[(n,np.round(np.quantile(np.concatenate([g['past_channel_raw'].reshape(-1,4),g['hold_channel_raw']]),[0,.05,.5,.95,1],axis=0),3)) for n,g in [('train',d['train']),('test',d['test']),('diag',i['campaign_diagnostics'])]])
