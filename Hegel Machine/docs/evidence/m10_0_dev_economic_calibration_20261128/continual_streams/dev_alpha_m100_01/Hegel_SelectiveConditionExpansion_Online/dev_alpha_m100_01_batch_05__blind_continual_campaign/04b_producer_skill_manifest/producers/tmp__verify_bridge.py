import sys,pickle, numpy as np
sys.path.insert(0,'/holdout_task')
import upstream_transducer as b
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X,Y=map(np.asarray,(C['true_channel_loads'],C['bridge_raw'])); i=np.arange(len(X)); clean=np.max(abs(Y-np.median(Y,0)),1)<5
f,c=b._fit(X,Y); print('fit',f,c); print('min slope',c[:,1]-abs(c[:,2]*f[0]),'contraction bounds',(abs(c[:,3])+abs(c[:,4]*f[1]))/(c[:,1]-abs(c[:,2]*f[0])))
e=np.full_like(Y,np.nan); z=np.full_like(Y,np.nan)
for j in range(5):
 m=i%5==j; f,c=b._fit(X[~m],Y[~m]); k=m&clean
 e[k]=(b._design(X[k],f)*c).sum(-1)-Y[k]; z[k]=b._invert(Y[k],f,c)-X[k]
for name,k in [('all',clean),('axes',clean&(i<16)),('mixed',clean&(i>=16)),('extreme',clean&(np.max(abs(X),axis=1)>1.45))]:
 print(name,'forward RMSE',np.sqrt(np.mean(e[k]**2,0)),'inverse RMSE',np.sqrt(np.mean(z[k]**2,0)),'inverse max',np.max(abs(z[k]),0))
f,c=b._fit(X,Y); O=b.calibrate_upstream(D)
for name in ('train','test'):
 for field in ('hold_channel_raw','past_channel_raw'):
  v=O[name][field]; err=np.max(abs((b._design(v,f)*c).sum(-1)-D[name][field])); print(name,field,'residual',err,'range',v.reshape(-1,4).min(0),v.reshape(-1,4).max(0)); assert err<1e-8
