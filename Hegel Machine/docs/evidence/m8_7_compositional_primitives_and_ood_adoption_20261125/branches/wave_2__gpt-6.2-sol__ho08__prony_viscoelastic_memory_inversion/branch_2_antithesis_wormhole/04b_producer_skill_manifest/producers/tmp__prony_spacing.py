import pickle
import numpy as np
from scipy.optimize import least_squares
from prony_probe import basis,solve
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); x=p['experiment']; t=x['train_t']; e=x['train_strain']; y=x['train_stress_raw']; d=np.column_stack((np.ones_like(t),t/t[-1]-.5)); tt=x['test_t']; et=x['test_strain']; gt=p['gt_test_stress']; grid=np.geomspace(.02,20,100); a=basis(t,e,grid,'ramp')

def fit(mask,kind,lam=0):
    c,b,r,s=solve(a[mask],d[mask],y[mask]); r=y-a@c-d@b; good=mask&(abs(r)<6*s)
    logtau=np.log([.18,1.4,8]); v0=np.r_[max(c[0],.1),30,16,9,b,logtau if kind=='free' else logtau[[0,2]]]
    low=np.r_[np.zeros(4),[-np.inf]*2,np.full(len(v0)-6,np.log(.02))]; high=np.r_[np.full(6,np.inf),np.full(len(v0)-6,np.log(20))]
    def times(v):
        z=v[6:]; return np.exp(z if kind=='free' else [z[0],(z[0]+z[1])/2,z[1]])
    def predict(v,T,E): return basis(T,E,times(v),'ramp')@v[:4]
    def fun(v):
        rr=(predict(v,t,e)+d@v[4:6]-y)[good]/s
        if lam: rr=np.r_[rr,lam*(v[6]-2*v[7]+v[8])]
        return rr
    f=least_squares(fun,v0,bounds=(low,high),max_nfev=200,ftol=1e-10); v=f.x
    return v,times(v),y-predict(v,t,e)-d@v[4:6],predict(v,tt,et),s,np.linalg.svd(f.jac,compute_uv=False)[-1]

for kind,lam in [('free',0),('free',1),('free',3),('free',10),('geometric',0)]:
    v,tau,r,yp,s,sv=fit(np.ones(len(t),bool),kind,lam)
    i=abs(r)<6*s
    print(kind,lam,'G',round(v[0],4),'tau',tau.round(4),'trainRMS',np.sqrt(np.mean(r[i]**2)),'testRRMSE',np.linalg.norm(yp-gt)/np.linalg.norm(gt),'minSV',sv)
    out=[]; gs=[]
    for lo,hi in [(0,1),(1,2),(2,4),(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,31)]:
        test=(t>=lo)&(t<hi); vv,_,rr,_,ss,_=fit(~test,kind,lam); out.append(round(np.sqrt(np.mean(rr[test&(abs(rr)<6*ss)]**2)),5)); gs.append(round(vv[0],3))
    print('blocked RMS',out,'blocked G',gs)
