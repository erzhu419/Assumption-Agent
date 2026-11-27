import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];tr=d['train'];te=d['test'];c=i['campaign_diagnostics'];dt=d['past_dt'];t=d['eval_times'];cal=d['calibration_load_cells'];raw=cal['bridge_raw'];y=cal['true_channel_loads'];b=m._bridge(cal)
print('bridge',b,'raw range',np.min(raw,0),np.max(raw,0)); print('sweeps',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in cal['single_channel_sweeps'].items()])
from itertools import combinations_with_replacement
for degree in (1,2,3):
 A=np.column_stack([np.ones(len(raw))]+[np.prod(raw[:,j],axis=1) for j in range(1,degree+1) for j in combinations_with_replacement(range(4),j)])
 pred=np.zeros_like(y)
 for q in np.array_split(np.random.default_rng(5).permutation(len(y)),5):
  z=np.setdiff1d(np.arange(len(y)),q);pred[q]=A[q]@np.linalg.lstsq(A[z],y[z],rcond=None)[0]
 e=pred-y;print('bridge OOF degree',degree,'channel rmse',np.sqrt(np.mean(e**2,0)),'stress rmse',np.sqrt(np.mean(e.sum(1)**2)))
x,xt=m._features(tr,te);p,h=m._loads(tr,b);pc,hc=m._loads(c,b);tau,beta=m._fit(x,p,h,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,beta)
_,xc=m._features(dict((k,np.concatenate([tr[k],te[k]],axis=0)) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')),c)
from scipy.linalg import lstsq
Y=np.column_stack((c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs'])); H=hc; P=pc; U=np.ones_like(H); labs=c['lab_id']; strata=c['stratum']; n=len(H)
xx=m._features(c,c)[1]; D=beta[2]+xx@beta[3:]; W=np.column_stack((U,H,D,P,xx)); age=np.cumsum(dt[::-1])[::-1]-dt
rms=lambda e: (np.sqrt(np.mean(e[:,:-1]**2)), np.sqrt(np.mean(e[:,-1]**2)))
def design(ta,kind):
    z=P@(np.exp(-age[:,None]/ta)*-np.expm1(-dt[:,None]/ta)); F=np.vstack((np.column_stack((-np.expm1(-t[:,None]/ta),t)),np.r_[1/ta,1.])); q=np.empty((n,0)); A=np.zeros((n,3)); G=[q,q,q];
    if kind.startswith('reset'): A[:,:2]=H[:,None]*beta[:2]; G[:2]=[[U,D],[U,D]]; G[2]=[U]
    if kind.startswith('free'): G[:2]=[[U,H,D],[U,H,D]]; G[2]=[U]
    if kind.startswith('retain'): G[:2]=[[U,H,D,z[:,0]],[U,H,D,z[:,1]]];G[2]=[U]
    if kind.startswith('past'): G[:2]=[[U,H,D,*P.T],[U,H,D,*P.T]]; G[2]=[U]
    if kind.startswith('pair'): G[:2]=[[U,H,D,*z.T],[U,H,D,*z.T]]; G[2]=[U]
    if '+' in kind:
        opt=kind.split('+')[1]; g={'sq':[D**2], 'hd':[H*D], 'feat':[*xx.T], 'cube':[D**2,D**3], 'zD':[*((z*D[:,None]).T)], 'lab':[labs==l for l in range(1,4)], 'geom':[np.mean(P,axis=1),P[:,-1]]}[opt]; G[0]=[*G[0],*g];G[1]=[*G[1],*g]
    return np.concatenate([np.column_stack(g)[:,None,:]*F[None,:,j,None] for j,g in enumerate(G)],axis=2),A@F.T, list(map(len,G))
def run(kind,idx,jdx,mode):
    def project(ta):
        K,B,L=design(ta,kind); coef=lstsq(K[idx].reshape(-1,sum(L)),(Y-B)[idx].ravel())[0]; return (K@coef+B-Y),np.split(coef,np.cumsum(L)[:-1])
    ta=tau if mode==0 else np.exp(least_squares(lambda z:project(np.exp(z))[0][idx].ravel(),np.log(tau),bounds=(np.log([.5,1.6]),np.log([1.0,20.0])),max_nfev=35).x)
    e,co=project(ta); return e[jdx],co,ta
allidx=np.arange(n); masks=[labs==a for a in np.unique(labs)]+[strata==a for a in np.unique(strata)]
for k in ['reset','free','retain','pair','past','free+sq','free+hd','free+feat','free+cube','free+lab','retain+zD','free+geom']:
 for mode in (0,1):
    e,co,ta=run(k,allidx,allidx,mode); folds=[rms(run(k,allidx[~w],allidx[w],mode)[0]) for w in masks]; print(k,mode,'times',ta,'full',np.round(rms(e),5),'worst',np.max(folds,axis=0).round(5),'coef',[p.round(4) for p in co]);
 if k in ('free','free+feat'):
    e,_,_=run(k,allidx,allidx,1); print('initial residual by coupon',np.round(np.column_stack((labs,strata,H,D,e[:,-1],np.isnan(c['coupon_features_raw']).any(axis=1))),4))
