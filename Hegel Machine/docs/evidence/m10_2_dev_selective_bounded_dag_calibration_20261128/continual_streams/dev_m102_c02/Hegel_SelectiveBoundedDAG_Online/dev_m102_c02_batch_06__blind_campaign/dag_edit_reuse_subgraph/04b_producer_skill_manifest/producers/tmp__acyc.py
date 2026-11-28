import numpy as np,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_a']['u']; y=s['panel_a']['v']; ok=np.max(abs(y-x),axis=1)<3; x=x[ok];y=y[ok];print('A rows kept',len(x))
for j in range(4):
 a,b=x[:,j],x[:,(j+1)%4]; v=y[:,j]; d=np.c_[a*0+1,a,b,a*b,a*a,a**3,a*a*b,a*b*b,b*b,b**3,a**3*b,a**4];
 for ix in ([0,1,2,3],[0,1,2,3,4,5],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6,7,8,9],list(range(12))):
  h=d[:,ix]; w=np.linalg.lstsq(h,v,rcond=None)[0]; r=h@w-v; print(j,ix, np.round(w,4),round(np.std(r),5),round(np.std(r/(1-(h*np.linalg.pinv(h).T).sum(1))),5))
