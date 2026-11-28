from batch_cv import s,x,y,z,folds
from batch_solver import _linear, _rbf, _predict
from scipy.linalg import cho_factor,cho_solve
import numpy as np

def predictions(tr,te):
    q=x[tr]/x[tr].std(0); qt=x[te]/x[tr].std(0)
    h,ht=_linear(q),_linear(qt)
    ans=[]
    for length in (1.25,1.55,1.8,2.,2.3,2.65,3.1):
        k,c=_rbf(q,q,length),_rbf(qt,q,length)
        for nugget in (1e-5,.0001,.001,.004):
            cf=cho_factor(k+nugget*np.eye(len(q)))
            vi=cho_solve(cf,np.eye(len(q))); a=vi@h
            p=vi-a@np.linalg.solve(h.T@a,a.T)
            press=np.mean(abs((p@y[tr])/p.diagonal()))
            yp=_predict(k,c,h,ht,y[tr,None],nugget)[:,0]
            ans.append((press,yp,(length,nugget)))
    return ans

outer=[predictions(tr,te) for tr,te in folds]
for name,choose in [('baseline',lambda a:next(p for _,p,k in a if k==(2.,.001))),('inner LOO winner',lambda a:min(a,key=lambda t:t[0])[1]),('inner LOO blend',lambda a:np.average(np.array([p for _,p,_ in a]),axis=0,weights=np.exp(-np.array([v for v,_,_ in a])/.0025)))]:
    e=np.array([np.mean(abs(choose(a)-y[te])) for a,(_,te) in zip(outer,folds)])
    print(name,np.round(e,5),'worst',round(e.max(),5))
print('inner winners',[min(a,key=lambda t:t[0])[2] for a in outer])
