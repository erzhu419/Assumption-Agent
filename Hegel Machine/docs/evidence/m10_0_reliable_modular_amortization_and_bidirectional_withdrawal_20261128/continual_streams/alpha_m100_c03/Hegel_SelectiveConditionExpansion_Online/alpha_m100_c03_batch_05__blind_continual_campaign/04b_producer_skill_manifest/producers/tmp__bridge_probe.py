import pickle,itertools,numpy as np
xdata=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=xdata['calibration_load_cells']; x0,y0=map(c.get,['true_channel_loads','bridge_raw']); ok=np.max(abs(y0),1)<5; x=x0[ok]; y=y0[ok]; print('removed',np.flatnonzero(~ok)); print('clean axial observations'); print(np.round(np.c_[x[:sum(np.count_nonzero(x,axis=1)<=1)],y[:sum(np.count_nonzero(x,axis=1)<=1)]],6)); print('clean ranges',x.min(0),x.max(0),y.min(0),y.max(0));
for deg in range(1,7):
 for additive in [True,False]:
  if not additive and (deg==1 or deg>4):continue
  terms=[()] + [t for k in range(1,deg+1) for t in itertools.combinations_with_replacement(range(4),k) if not additive or t[0]==t[-1]]
  A=np.stack([np.prod(x[:,t],axis=1) for t in terms],1); U,s,V=np.linalg.svd(A,full_matrices=False); coef=np.linalg.lstsq(A,y,rcond=None)[0]; r=y-A@coef; loo=r/np.maximum(1e-8,1-(U**2).sum(1))[:,None]; print(deg,additive,len(terms),'RMS',np.round(np.mean(r*r,0)**.5,5),'LOO RMS/worst',np.round(np.mean(loo*loo,0)**.5,5),np.round(np.max(abs(loo),0),5));
  if deg in (2,3) and not additive: print([(t,list(row)) for t,row in zip(terms,coef.round(4))]);
