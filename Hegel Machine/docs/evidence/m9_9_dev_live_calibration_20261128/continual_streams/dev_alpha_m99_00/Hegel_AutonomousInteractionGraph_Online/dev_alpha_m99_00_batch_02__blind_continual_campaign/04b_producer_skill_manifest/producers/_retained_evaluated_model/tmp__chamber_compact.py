import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_structure import *

D=theta[2]+old._features(g)@theta[3:]
U=np.ones(len(h)); full=np.ones(len(h),bool)
folds=[(col,int(v),g[col]==v) for col in ('lab_id','stratum') for v in np.unique(g[col])]

def fit_compact(train,mode,D,weight_rate=2.):
    def matrices(q):
        s,f=np.exp(q[:2]); fast,slow=s,s+f
        zf=memory(p,dt,fast); zs=memory(p,dt,slow)
        m0=memory(p,dt,np.exp(q[2])) if len(q)>2 else zf
        m1=memory(p,dt,np.exp(q[3])) if len(q)>3 else zs
        if mode=='intrinsic':
            m0,m1=z.T
        m={'intrinsic':([h,m0,m1,D,U],)*3,
           'retimed':([h,m0,m1,D,U],)*3,
           'free_memory':([h,m0,m1,D,U],)*3,
           'one_memory':([h,m0,D,U],)*3,
           'gap_hold':([h-m0,D],[h,m0,D],[h,m0,D]),
           'gap_pair':([h-m0,D],[h,m1,D],[h,m1,D]),
           'gap_affine':([h-m0,D,U],[h,m1,D,U],[h,m1,D,U]),
           'compact':([h-m0,D],[h,m0,m1,D],[h,m0,m1,D]),
           'compact_bias':([h-m0,D,U],[h,m0,m1,D,U],[h,m0,m1,D,U])}[mode]
        b=basis(np.array([fast,slow])); return np.concatenate([np.column_stack(a)[:,None,:]*b[None,:,j,None] for j,a in enumerate(m)],axis=2)
    w=np.r_[np.ones(len(t)),weight_rate]; yy=y[train]*w
    def calc(q):
        v=matrices(q); a=v[train]*w[None,:,None]; c=np.linalg.lstsq(a.reshape(-1,v.shape[2]),yy.ravel(),rcond=None)[0]; return v@c,c
    times=[.49,.96]+({'free_memory':[.5,2.], 'one_memory':[.5]}.get(mode,[]))
    q=least_squares(lambda q:((calc(q)[0][train]-y[train])*w).ravel(),np.log(times),bounds=(np.log(np.array([.25,.15]+[.15]*(len(times)-2))),np.log(np.array([.85,9.]+[15.]*(len(times)-2)))),max_nfev=100).x
    pred,c=calc(q); ts=np.exp(q);ts[1]+=ts[0]
    return pred,c,ts

if __name__=='__main__':
 for m in ['intrinsic','retimed','free_memory','one_memory','gap_hold','gap_pair','gap_affine','compact','compact_bias']:
    pred,co,ts=fit_compact(full,m,D); e=pred-y
    scores=[]; pool=[]
    for col,v,ix in folds:
        r=fit_compact(~ix,m,D)[0][ix]-y[ix];scores.append([rm(r[:,:-1]),rm(r[:,-1])]);pool.append(r)
    pool=np.concatenate(pool); print(m,'times',np.round(ts,5),'fit',np.round([rm(e[:,:-1]),rm(e[:,-1])],5),'OOF',np.round([rm(pool[:,:-1]),rm(pool[:,-1])],5),'worst',np.round(np.max(scores,0),5),'coef',np.round(co,5).tolist())
 print('test joint-vs-test-only drift RMSE/max',rm((old._features(tr,te)[n:]-old._features(te))@theta[3:]),np.max(np.abs((old._features(tr,te)[n:]-old._features(te))@theta[3:])))
 for name,X in [('train-only',old._features(tr)),('joint',old._features(tr,te)[:n])]:
    tau2,theta2=old._fit(coupon(tr,X),dt,t,*obs(tr)); curve,rate=old._design(tau2,coupon(tr,X),dt,t); e=curve@theta2-obs(tr)[0]; er=rate@theta2-obs(tr)[1];print('train alignment',name,np.round(tau2,6),'curve median/90pct',np.round(np.quantile(np.abs(e),[.5,.9]),5),'rate RMSE',round(rm(er),5))
