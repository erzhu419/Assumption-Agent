exec(open('/tmp/m98_inspect.py').read().split('for g,x1')[0])
from scipy.optimize import least_squares
from itertools import product
G=diag; n=len(G['lab_id']); p=G['past_channel_raw']@cal[:-1]+cal[-1]; h=G['hold_channel_raw']@cal[:-1]+cal[-1]; y=np.c_[G['strain_curves_obs']-G['strain_0'][:,None],G['instant_rate_obs']]
keys=('lab_id','stratum','coupon_features_raw','ref_coupons')
def align(*groups):
    return m._aligned_features(groups[0],{k:np.concatenate([g[k] for g in groups[1:]]) for k in keys})[-n:]
def frozen():
    labs=np.r_[train['lab_id'],test['lab_id']]; refs=np.vstack([train['ref_coupons'],test['ref_coupons']]); a=G['coupon_features_raw'].copy()
    for lab in np.unique(G['lab_id']):
        v=refs[labs==lab]; a[G['lab_id']==lab]=(a[G['lab_id']==lab]-np.mean(v,0).mean())/np.median(v.std(0)) if False else (a[G['lab_id']==lab]-np.median(v.mean(0)))/np.median(v.std(0))
    ref=m._aligned_features(train,test)[:,1:]; raw=np.vstack([train['coupon_features_raw'],test['coupon_features_raw']]); _,idx,cnt=np.unique(np.c_[labs,np.r_[train['stratum'],test['stratum']]],axis=0,return_inverse=True,return_counts=True); wt=(1/cnt[idx])[:,None]; mask=np.isfinite(raw); fill=(wt*ref*mask).sum(0)/(wt*mask).sum(0)
    return np.c_[np.ones(n),np.where(np.isfinite(a),a,fill)]
alts={'joint':align(train,test,G),'frozen':frozen(),'train_diag':align(train,G),'diag':align({k:G[k][:0] for k in keys},G)}
T=np.r_[t,1.]
def expbasis(s):return np.r_[-np.expm1(-t/s),1/s]
def design(q,z,model):
    rate=z@w[2:]; cols=[]; b=[expbasis(l) for l in q]
    def add(f,v):cols.append(f[:,None]*v)
    if model.startswith('fact'):
        for j in range(2): add(h,b[j]); add(np.ones(n),b[j]) if 'inter' in model else None
        add(rate,b[2]); add(rate,T) if 'lin' in model else None
    else:
        for j in range(2):
            add(h,b[j]); add(rate,b[j]); add(np.ones(n),b[j]) if 'inter' in model else None
        add(rate,T); add(np.ones(n),T) if 'inter' in model else None
        if 'full' in model:
            for f in z[:,1:].T: add(f,b[1]); add(f,T)
        if 'mem' in model:
            state=np.zeros((n,2))
            for a,ps in zip(-np.expm1(-dt[:,None]/q[:2]),p.T): state+=a*(ps[:,None]-state)
            for j in range(2):add(state[:,j],b[j])
    return np.stack(cols,2)
def fit(z,model,free,ids):
    def calc(q,ret=False):
        A=design(q,z,model); F=A[ids].reshape(-1,A.shape[-1]); coef=np.linalg.lstsq(F,y[ids].ravel(),rcond=1e-9)[0]; e=A@coef-y
        return (e,coef) if ret else e[ids].ravel()
    q=tau.copy()
    if model.startswith('fact'):q=np.r_[q,1.2]
    if free:
        sols=[least_squares(lambda l:calc(np.exp(l)),np.log(init),bounds=(np.log(.05),np.log(30)),max_nfev=140) for init in ([q] if len(q)==2 else [q,[tau[0],tau[1],.3],[tau[0],tau[1],7]])]; q=np.exp(min(sols,key=lambda o:o.cost).x)
    return (*calc(q,True),q)
def metrics(e):return np.round([np.mean(e[:,:-1]**2)**.5,np.mean(e[:,-1]**2)**.5],5)
for an,z in alts.items():
    for model,free in [('mix',False),('mixinter',False),('mix',True),('mixinter',True),('mixmem',False),('mixfull',False),('factlin',True),('factinterlin',True)]:
        e,c,q=fit(z,model,free,np.ones(n,bool)); es=[]
        for lab in np.unique(G['lab_id']):
            ids=G['lab_id']!=lab; es.append(metrics(fit(z,model,free,ids)[0][~ids]))
        print(an,model,free,metrics(e),np.max(es,axis=0),np.round(q,4),np.round(c,4))
