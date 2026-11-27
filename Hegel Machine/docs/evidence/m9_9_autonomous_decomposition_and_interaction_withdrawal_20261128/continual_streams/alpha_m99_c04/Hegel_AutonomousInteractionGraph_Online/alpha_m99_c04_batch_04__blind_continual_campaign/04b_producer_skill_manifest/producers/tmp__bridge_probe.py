import pickle,itertools,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=p['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def powers(deg,kind):
 return np.array([v for v in itertools.product(range(deg+1),repeat=4) if sum(v)<=deg and (kind=='full' or np.count_nonzero(v)<=1 or sum(v)<=2)])
def features(a,p):return np.prod(a[:,None,:]**p[None,:,:],axis=-1)
def robust(a,b):
 w=np.ones(len(a))
 for _ in range(35):
  coef=np.linalg.lstsq(a*w[:,None],b*w[:,None],rcond=None)[0];r=np.linalg.norm(b-a@coef,axis=1); w=np.sqrt(np.minimum(1,max(.0003,np.median(r)*.6)/np.maximum(r,1e-15)))
 mask=r<max(.001,np.median(r)*5)
 if mask.sum()>=a.shape[1]+2:coef=np.linalg.lstsq(a[mask],b[mask],rcond=None)[0]
 return coef
for kind,d in [('full',1),('full',2),('add',3),('full',3),('add',4),('add',5),('add',6),('add',7)]:
 po=powers(d,kind); a=features(x,po); b=robust(a,y); r=np.linalg.norm(y-a@b,axis=1)
 pred=np.empty_like(y); rng=np.random.default_rng(82)
 for val in np.array_split(rng.permutation(len(x)),5):
  tr=np.setdiff1d(np.arange(len(x)),val);pred[val]=a[val]@robust(a[tr],y[tr]);
 cv=np.linalg.norm(pred-y,axis=1); good=np.argsort(r)[:-5]; print(kind,d,len(po),'fit q',np.quantile(r,[.5,.8,.9,.95]),'oof q excluding five',np.quantile(cv[good],[.5,.9,.975,1]));
 if d in (2,3,5) and kind!='add' or d==5:
  for v,beta in sorted(zip(po,b),key=lambda q:(sum(q[0]),list(q[0]))): print(v,beta)
  print('top ten residuals',[(int(i),round(r[i],5),round(cv[i],5)) for i in np.argsort(r)[-10:]])
print('axis raw incl all 16 axis rows:'); print(np.c_[x[:16],y[:16]])
