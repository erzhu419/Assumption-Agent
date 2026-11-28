import itertools as it
import pickle
import numpy as np
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for k,x in st.items():
    print(k, {a: np.shape(b) for a,b in x.items()} if isinstance(x,dict) else np.shape(x))
u,v=(st['panel_a'][k] for k in ('u','v'))
n,d=u.shape
print('u ranges',u.min(0),u.max(0),'v ranges',v.min(0),v.max(0),'test ranges',st['test']['x'].min(0),st['test']['x'].max(0))
terms=[()] + [(j,) for j in range(d)]
terms2=list(it.combinations_with_replacement(range(d),2)); terms3=list(it.combinations_with_replacement(range(d),3)); cubes=[(j,)*3 for j in range(d)]
def design(x,ts): return np.column_stack([np.prod(x[:,t],axis=1) for t in ts])
def fit(A,Y):
    w=np.ones(len(A))
    for scale in (.15,.08,.04,.02,.01,.005,.003,.003,.003,.003):
        b=np.linalg.lstsq(A*w[:,None],Y*w[:,None],rcond=None)[0]
        e=np.linalg.norm(Y-A@b,axis=1)
        w=(1+(e/scale)**2)**-.5
    keep=np.argsort(e)[:-3]
    return np.linalg.lstsq(A[keep],Y[keep],rcond=None)[0],keep
if __name__=='__main__':
 for name, ts in [('affine',terms),('pure_cubes',terms+cubes),('quadratic',terms+terms2),('quad+cubes',terms+terms2+cubes),('homogeneous_cubics',terms+terms3),('full',terms+terms2+terms3)]:
    A=design(u,ts); b,keep=fit(A,v)
    err=np.linalg.norm(A@b-v,axis=1); order=np.argsort(err)
    ee=[]
    if len(ts)<n-5:
        for j in range(n):
            ids=np.arange(n)!=j
            bj,_=fit(A[ids],v[ids]);ee.append(np.linalg.norm(A[j]@bj-v[j]))
    print(name,'p',len(ts),'median/worst retained',np.median(err),err[order[-4]],'outliers',order[-3:],err[order[-3:]],'leave-one-out quantiles',np.quantile(ee,[.5,.85,.9,.95]) if ee else None)
    if name in ('pure_cubes','quad+cubes','homogeneous_cubics','full'):
        for t,c in zip(ts,b):
            if max(abs(c))>.009: print(t,np.round(c,4))
