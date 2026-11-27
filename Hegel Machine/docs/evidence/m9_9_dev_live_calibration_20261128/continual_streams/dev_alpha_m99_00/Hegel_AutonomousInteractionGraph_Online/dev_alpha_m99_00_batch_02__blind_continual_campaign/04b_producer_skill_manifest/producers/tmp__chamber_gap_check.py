import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from chamber_compact import *
from itertools import product
D=theta[2]+old._features(g)@theta[3:]
g0,g1=(h[:,None]-z).T
U=np.ones(len(h)); full=np.ones(len(h),bool)
models={
 'affine': ((h,z[:,0],z[:,1],D,U),)*3,
 'two_gaps': ((g0,g1,D),)*3,
 'two_gaps_bias': ((g0,g1,D,U),)*3,
 'compact_gaps': ((g0,D),(g0,g1,D),(g1,D)),
 'compact_gaps_bias': ((g0,D,U),(g0,g1,D,U),(g1,D,U)),
 'mixed': ((g0,g1,D),(g0,g1,D),(g0,g1,D,U)),
}

def score(m,train,rate_w=2.):
    w=np.r_[np.ones(len(t)),rate_w]
    def calculate(q):
        times=np.exp(q); times[1]+=times[0]; b=basis(times)
        a=np.concatenate([np.column_stack(v)[:,None,:]*b[None,:,j,None] for j,v in enumerate(models[m])],axis=2)
        aw=a[train]*w[None,:,None]
        coef=np.linalg.lstsq(aw.reshape(-1,a.shape[2]),(y[train]*w).ravel(),rcond=None)[0]
        return a@coef,coef,times
    opt=least_squares(lambda q:((calculate(q)[0][train]-y[train])*w).ravel(),np.log([.49,.95]),bounds=(np.log([.25,.15]),np.log([.85,9.])),max_nfev=80)
    return calculate(opt.x)

quart=[('hold',lab,(h<=np.quantile(h,.25) if lab==0 else h>=np.quantile(h,.75))) for lab in (0,1)]
for m in models:
    p0,c,ts=score(m,full); ee=p0-y
    print(m, 'full', np.round([rm(ee[:,:-1]),rm(ee[:,-1])],6), 'times', np.round(ts,6), 'coef', np.round(c,5))
    for name,fs in [('lab/stratum',folds),('load tails',quart)]:
        es=[score(m,~ix)[0][ix]-y[ix] for _,_,ix in fs]; a=np.concatenate(es)
        print(name,'pooled',np.round([rm(a[:,:-1]),rm(a[:,-1])],6),'worst',np.round(np.max([[rm(e[:,:-1]),rm(e[:,-1])] for e in es],axis=0),6))
print('largest residual coupons: index/lab/stratum/missing/curve/rate')
p0,_,_=score('affine',full); e=p0-y
for i in np.argsort(np.linalg.norm(e,axis=1))[-5:]:
 print(i,g['lab_id'][i],g['stratum'][i],np.flatnonzero(np.isnan(g['coupon_features_raw'][i])),round(rm(e[i,:-1]),5),round(e[i,-1],5))
print('dynamic fit vs staged calibration')
fitted=calibrate_upstream(raw); print(*(np.max(abs(fitted[s]['hold_channel_raw']-d[s]['hold_channel_raw'])) for s in ('train','test')))
