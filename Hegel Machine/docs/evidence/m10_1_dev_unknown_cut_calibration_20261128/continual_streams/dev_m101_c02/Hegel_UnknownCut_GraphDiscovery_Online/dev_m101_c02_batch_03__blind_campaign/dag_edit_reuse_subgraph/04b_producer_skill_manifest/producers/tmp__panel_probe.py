import pickle, numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v'))
def features(x, degree, cross=True):
    inds=[i for d in range(1,degree+1) for i in combinations_with_replacement(range(x.shape[1]),d) if cross or len(set(i))==1]
    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for i in inds]), ['1']+[''.join(str(j) for j in i) for i in inds]
def robust(a,b):
    w=np.ones(len(a))
    for _ in range(150):
        c=np.linalg.lstsq(a*w[:,None],b*w[:,None],rcond=None)[0]
        r=np.linalg.norm(b-a@c,axis=1)
        w=np.sqrt(np.minimum(1, max(1e-10, .08*np.median(r))/np.maximum(r,1e-15)))
    good=np.argsort(r)[:len(r)-3]; c=np.linalg.lstsq(a[good],b[good],rcond=None)[0]
    return c,np.linalg.norm(b-a@c,axis=1),good
print('ranges',u.min(0),u.max(0),s['test']['x'].min(0),s['test']['x'].max(0))
for degree,cross in [(1,True),(2,False),(2,True),(3,False),(3,True)]:
    a,names=features(u,degree,cross); c,r,g=robust(a,v); cv=[]
    for i in g:
        m=np.arange(len(u))!=i; cc,rr,gg=robust(a[m],v[m]); cv.append(np.linalg.norm(v[i]-a[i]@cc))
    print(degree,cross,'median/cleanmax/outliers',np.median(r),max(r[g]),np.argsort(r)[-3:],r[np.argsort(r)[-3:]],'LOO median/max',np.median(cv),max(cv));
    if degree<=2 and cross: print(list(zip(names,np.round(c,5).tolist())))
