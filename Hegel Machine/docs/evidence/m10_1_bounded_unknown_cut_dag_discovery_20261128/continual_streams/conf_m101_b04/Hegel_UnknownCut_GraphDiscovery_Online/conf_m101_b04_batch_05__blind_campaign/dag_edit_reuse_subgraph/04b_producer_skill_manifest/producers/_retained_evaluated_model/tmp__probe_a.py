import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; n,d=u.shape
terms=[t for k in range(4) for t in combinations_with_replacement(range(d),k)]
for label,ts in [('affine',terms[:5]),('diagonal cubic',terms[:5]+[(i,i,i) for i in range(4)]),('odd cubic',[t for t in terms if len(t)!=2]),('full cubic',terms)]:
 a=np.column_stack([np.prod(u[:,t],axis=1) for t in ts]); w=np.ones(n)
 for j in range(50):
  c=np.linalg.lstsq(a*w[:,None],v*w[:,None],rcond=None)[0]; e=np.linalg.norm(a@c-v,axis=1); w=1/np.sqrt(1+(e/max(.014,np.median(e)))**2)
 keep=np.argsort(e)[:-3]; c=np.linalg.lstsq(a[keep],v[keep],rcond=None)[0]; e=np.linalg.norm(a@c-v,axis=1); q=np.linalg.qr(a[keep])[0]; loo=e[keep]/np.maximum(1-(q*q).sum(1),1e-8)
 print(label, 'outliers', np.argsort(e)[-3:], np.sort(e)[-5:], 'clean RMSE',np.sqrt(np.mean(e[keep]**2)), 'LOO rms/max',np.sqrt(np.mean(loo**2)),max(loo)); print([(t,np.round(row,3).tolist()) for t,row in zip(ts,c)])
print('telemetry',s['telemetry_vec'],'summary',s['train']['summary'])
