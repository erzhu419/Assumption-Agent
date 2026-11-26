import pickle
import numpy as np
from alpha_ep01_solver import _aligned,_physical,_Design,_fit,_mad,_crossing
p=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; lab,tl=tr['lab_id'],te['lab_id']; n=len(lab); t=d['eval_times']; x,c=_aligned(d)
a,b=(_Design(*_physical(d,q),d['past_dt'],t,f,q['initial_memory_states']) for q,f in [(tr,x[:n]),(te,x[n:])]); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
# Differences remove whole-curve measurement offsets before estimating slope.
j,k=np.triu_indices(len(t),1); q=t[k]-t[j]>1.5; j,k=j[q],k[q]
def slope(r): return np.median((r[:,k]-r[:,j])/(t[k]-t[j]),axis=1)
def rmse(r): return np.sqrt(np.mean(r*r))
def predict(lt,w,keep,z,ls,features,full,rate):
    train=a.matrix(lt)@w; ds=slope(y[:,:-1]-train[:,:-1]); rs=y[:,-1]-train[:,-1]; base=z.matrix(lt)@w
    prior=np.zeros(len(ls)); dev=np.zeros(len(ls)); bias=np.zeros(len(ls))
    for l in np.unique(ls):
        i=(lab==l)&keep; o=ls==l
        if not i.any():continue
        f=x[:n][i]@w[3:]; fc=np.median(f); s=np.column_stack((np.ones(i.sum()),f-fc));
        # A reference-panel offset shifts drift; a scale error changes its feature-dependent part.
        theta=np.linalg.lstsq(s,np.clip(ds[i],np.median(ds[i])-.04,np.median(ds[i])+.04),rcond=None)[0]
        prior[o]=np.median(ds[i]); dev[o]=theta[0]+theta[1]*(features[o]@w[3:]-fc)
        bias[o]=np.median(rs[i]-ds[i])
    raw=rate-base[:,-1]; gain=np.where(full,.15,1.)
    return {key:base[:,:-1]+change[:,None]*t for key,change in {'base':gain*raw,'offset':gain*raw+(1-gain)*prior,'affine':gain*raw+(1-gain)*dev,'affine_ratebias':gain*(raw-bias)+(1-gain)*dev}.items()}
fold=np.empty(n,dtype=int)
for l in np.unique(lab):
    for s in np.unique(tr['stratum']):
        ids=np.flatnonzero((lab==l)&(tr['stratum']==s)); fold[ids]=np.arange(len(ids))%4
cv={};
for jfold in range(4):
    keep=(fold!=jfold)&c[:n]; lt,w=_fit(a,y,keep); residual=y[:,:-1]-(a.matrix(lt)@w)[:,:-1]; held=fold==jfold
    for key,pred in predict(lt,w,keep,a,lab,x[:n],c[:n],y[:,-1]).items():
        cv.setdefault(key,np.empty(n))[held]=slope(y[:,:-1]-pred)[held]*t[-1]
for key,e in cv.items():
    print('out-of-fold',key,'endpoint-proxy RMSE by lab complete/missing',[[round(rmse(e[(lab==l)&s]),5) for s in (c[:n],~c[:n])] for l in np.unique(lab)])
lt,w=_fit(a,y,c[:n]);
for name,z in predict(lt,w,c[:n],b,tl,x[n:],c[n:],te['instant_rate_obs']).items():
    initial=te['strain_0']; curve=initial[:,None]+z
    es=[curve-g['te_clean'],z[:,-1]/np.maximum(abs(b.hold),.5)-g['te_secant'],_crossing(t,initial,curve,te['critical_strain'])-g['te_cross']]
    print('public',name,'RMSE',np.round(list(map(rmse,es)),6),'worst lab',np.round([max(rmse(e[tl==l]) for l in np.unique(tl)) for e in es],6),'secant worst complete/missing',np.round([max(rmse(es[1][(tl==l)&s]) for l in np.unique(tl) if ((tl==l)&s).any()) for s in (c[n:],~c[n:])],6))
print('calibration stress RMSE',rmse(b.hold-g['te_hold_true']),rmse(b.past-g['te_hist_true']))
