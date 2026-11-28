import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d):
 idx=[p for k in range(d+1) for p in itertools.combinations_with_replacement(range(4),k)]
 return np.array([np.prod(x[:,p],axis=1) for p in idx]).T,idx
for panel,key in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[panel]; x,y=p['u'],p[key]; keep=np.ones(len(x),bool)
 if panel=='panel_a':
  z,_=poly(x,1)
  for _ in range(3):
   r=np.linalg.norm(z@fit(z[keep],y[keep])-y,axis=1); keep=r<3*np.median(r)
 print(panel,key,'kept',keep.sum())
 for d in [1,2,3]:
  z,idx=poly(x,d); xx=z[keep]; yy=y[keep]; c=fit(xx,yy); e=xx@c-yy; h=np.sum(xx*np.linalg.pinv(xx).T,axis=1); print('degree',d,'rms',np.std(e),'LOO',np.std(e/np.maximum(1-h,.0001)[:,None]) if xx.shape[0]>xx.shape[1] else None)
  if panel=='panel_a' or (d==2 and key=='v2'):
   for j in range(4): print(j,[(idx[i],round(c[i,j],5)) for i in np.argsort(abs(c[:,j]))[-7:][::-1]])
