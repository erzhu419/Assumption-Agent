from shape04 import *
from itertools import product
order=np.random.default_rng(194).permutation(len(w)); edge=np.argsort(abs(w).max(1))[-16:]; folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(order,j),j) for j in np.array_split(order,5)]+[(np.setdiff1d(order,edge),edge)]
def time_fit(pred, train):
    sig=pred(w[train])[1]; _,_,vt=np.linalg.svd(sig,full_matrices=False)
    def basis(p,kind):
        a,b=p
        if kind==0:return np.c_[np.log1p(a*times),-np.expm1(-b*times)]
        if kind==1:return np.c_[-np.expm1(-a*times),-np.expm1(-b*times)]
        if kind==2:return np.c_[np.log1p(a*times),np.log1p(b*times)]
    def fit(p,kind):
        B=basis(p,kind); return B@np.linalg.pinv(B)
    options=[]
    for kind in range(3):
        for p in ([.5,1.8],[2,6]):
            o=least_squares(lambda p:(sig@fit(p,kind)-sig).ravel(),p,bounds=(.005,30),max_nfev=50)
            options.append((o.cost,kind,o.x,fit(o.x,kind)))
    return min(options,key=lambda o:o[0]),vt[:2].T@vt[:2]

def shape(name,q,p):
    if name=='power harmonic': return q**3/(1+p[0]*abs(q)**p[1])+p[2]*q+p[3]*np.sin(p[4]*q)
    if name=='quartic harmonic': return q**3/(1+p[0]*q*q+p[1]*q**4)+p[2]*q+p[3]*np.sin(p[4]*q)
    if name=='rational double': return q**3/(1+p[0]*q*q)+p[1]*q+p[2]*np.sin(p[3]*q)+p[4]*np.sin(p[5]*q)
    return q**3+p[0]*q+p[1]*q**5+p[2]*q**7+p[3]*np.sin(p[4]*q)

for name,p,lo,hi in [('power harmonic',[.5,2,-.08,.01,11],[0,.5,-1,-.3,3],[5,12,1,.3,28]),('quartic harmonic',[.4,.1,-.08,.01,11],[0,0,-1,-.3,3],[5,5,1,.3,28]),('seventh harmonic',[-.08,-.28,0,.01,11],[-1,-2,-1,-.3,3],[1,1,1,.3,28]),('rational double',[.6,-.08,.01,10,.005,16],[0,-1,-.3,3,-.3,3],[5,1,.3,28,.3,30])]:
    for alpha in (.7,4):
        es=[]; pars=[]
        for i,j in folds:
            opts=[]
            for a in (7,11,14,19):
                p1=p.copy(); ix=3 if name=='rational double' else 4; p1[ix]=a
                if name=='rational double':p1[5]=a+5
                pred,fit,r=solve_shape(lambda q,p:shape(name,q,p),p1,(lo,hi),i,alpha); opts.append((r@r,pred,fit,r))
            _,pred,fit,r=min(opts,key=lambda o:o[0]); es.append(scores(i,j)(pred(w[j]))); pars.append(fit)
        print(name,alpha,np.round(pars[0],5),'56/16',es[0],'worst',np.max(es,axis=0),'edge',es[-1],flush=True)
for name,f,p,b in __import__('knee04').forms2():
    if name not in ('odd cheb 15',): continue
    es=[]
    for i,j in folds:
        pred,fit,r=solve_shape(f,p,b,i,4); yy,tt=pred(w[j]); tf,svd=time_fit(pred,i); es.append([abs(tt-t[j]).mean(),abs(tt@tf[-1]-t[j]).mean(),abs(tt@svd-t[j]).mean()]); print('temporal fit',tf[:3],flush=True)
    print('trajectory raw, physical rank2, SVD rank2','56/16',es[0],'worst',np.max(es,axis=0),'edge',es[-1])
