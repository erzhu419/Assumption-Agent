import pickle,numpy as np,itertools
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=145)
print('keys',D.keys());
for k,p in D.items():
 if not isinstance(p,dict): print(k,p); continue
 for n,v in p.items():
  a=np.asarray(v); print(k,n,a.shape,'mean',a.mean(axis=0),'std',a.std(axis=0),'minmax',a.min(),a.max(),'head',a[:3])
 if 'u' not in p: continue
 u=p['u']; one=np.ones(len(u)); x=np.column_stack((one,u)); quad=np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
 for n,y in p.items():
  if n=='u':continue
  print('FIT',k,n)
  for label,f in [('affine',x),('quadratic',np.column_stack((x,quad))),('separable-cubic',np.column_stack((x,quad,u**3))),('sin+tanh',np.column_stack((x,np.sin(u),np.tanh(u)))),('separable-nonlinear',np.column_stack((x,np.sin(u),np.cos(u),np.tanh(u),u**2)))]:
   coef=np.linalg.lstsq(f,y,rcond=None)[0]; q=np.linalg.qr(f)[0]; r=y-f@coef; lev=(q*q).sum(1); print(label,'rmse',np.sqrt((r*r).mean(0)),'loo',np.mean(abs(r/np.maximum(1-lev[:,None],1e-10)),axis=0));
   if label=='affine':print('coef',coef,'largest row residuals',np.sort(np.linalg.norm(r,axis=1))[-8:]);
