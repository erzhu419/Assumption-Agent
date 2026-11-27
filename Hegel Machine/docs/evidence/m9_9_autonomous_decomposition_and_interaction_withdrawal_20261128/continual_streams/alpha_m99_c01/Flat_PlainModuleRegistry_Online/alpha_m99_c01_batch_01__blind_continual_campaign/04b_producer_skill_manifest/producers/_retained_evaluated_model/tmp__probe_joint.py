exec(open('/tmp/probe_chamber.py').read().split('for tag,')[0])
Y=np.c_[yc,rc]; n=len(Y); q=np.r_[t,1.]; r=beta[2]+xc@beta[3:]; age=np.cumsum(dt[::-1])[::-1]-dt

def design(u,names):
    z=pc@(np.exp(-age[:,None]/u)*-np.expm1(-dt[:,None]/u)); g=hc[:,None]-z
    T=np.c_[-np.expm1(-np.r_[t,0.][:,None]/u),q]; T[-1,:2]=1/u
    ds={'1':np.ones(n),'h':hc,'z0':z[:,0],'z1':z[:,1],'g0':g[:,0],'g1':g[:,1],'r':r,'p':pc,'x':xc}
    return np.concatenate([np.einsum('ni,k->nki',np.column_stack([ds[v] for v in names[j].split()]),T[:,j]) for j in range(3) if names[j]],axis=2)

def lin(A,ids):
    X=A[ids].reshape(-1,A.shape[-1]); return np.linalg.lstsq(X,Y[ids].ravel(),rcond=None)[0]

models=[('h r','h r',''),('h r','g1 r',''),('h r','g0 r',''),('h r','1 h z0 z1 r',''),('1 h z0 z1 r','1 h z0 z1 r',''),('h r','g1 r','1 r'),('1 h z0 z1 r','1 h z0 z1 r','1 r'),('1 h p x','1 h p x','1 h p x'),('1 h p r','1 h p r',''),('1 h p r','1 h p r','1 r')]
for names in models:
    ids=np.arange(n)
    for flexible in (False,True):
        def fit(ii):
            def residual(lu):
                A=design(np.exp(lu),names); return (A[ii]@lin(A,ii)-Y[ii]).ravel()
            u=np.exp(least_squares(residual,np.log(tau),bounds=(np.log([.3,1.5]),np.log([1.5,15.])),max_nfev=45).x) if flexible else tau
            A=design(u,names); return A,lin(A,ii),u
        A,b,u=fit(ids); e=A@b-Y; oo=np.zeros_like(Y)
        for lab in np.unique(d['lab_id']):
            mask=d['lab_id']==lab; F,c,_=fit(ids[~mask]); oo[mask]=F[mask]@c-Y[mask]
        print(names,flexible,'tau',np.round(u,5),'b',np.round(b,4),'fit',np.round(np.sqrt(np.mean(e**2,axis=0))[[0,-2,-1]],5),'LOLabO worst',np.round(np.max([np.sqrt(np.mean(oo[d['lab_id']==l]**2,axis=0)) for l in np.unique(d['lab_id'])],axis=0)[[0,-2,-1]],5))
print('batch items',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
import json
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print('log',str({k:v for k,v in row.items() if k!='facility_telemetry_trace'})[:1800])
