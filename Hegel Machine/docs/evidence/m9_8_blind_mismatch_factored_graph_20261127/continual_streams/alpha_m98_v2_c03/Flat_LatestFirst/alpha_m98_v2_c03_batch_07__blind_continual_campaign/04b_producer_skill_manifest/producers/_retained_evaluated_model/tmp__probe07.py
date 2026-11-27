import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_06_solver.py'; spec=importlib.util.spec_from_file_location('base',p); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr,te=d['train'],d['test']; t=np.asarray(d['eval_times']); dt=np.asarray(d['past_dt']); c=d['calibration_load_cells']; load=c['true_channel_loads']; raw=c['bridge_raw']; a=b._bridge_basis(load)
def qtile(x): return np.round(np.quantile(np.abs(x),[.5,.8,.95,1]),6)
def target(z): return np.column_stack((z['strain_curves_obs']-z['strain_0'][:,None],z['instant_rate_obs']))
for width in [5,13]:
 A=a[:,:width]; e=[]; es=[]
 for fold in range(5):
  test=np.arange(len(A))%5==fold; B=A[~test]; R=raw[~test]; q=np.linalg.lstsq(B,R,rcond=None)[0]
  for _ in range(4):
   er=np.max(abs(B@q-R),axis=1); clean=er<=max(6*np.median(er),.002); q=np.linalg.lstsq(B[clean],R[clean],rcond=None)[0]
  err=A[test]@q-raw[test]; e.extend(np.max(abs(err),axis=1)); es.extend(err@np.linalg.solve(q[1:5],np.ones(4)))
 print('bridge width',width,'channel max quantiles',qtile(e),'approx total quantiles',qtile(es))
 def stress(raw): return b._stress(raw,np.pad(q,((0,13-width),(0,0))))
 X=b._features(tr,te); n=len(tr['lab_id']);
 def coupon(z,x): return (stress(z['past_channel_raw']),stress(z['hold_channel_raw']),np.column_stack((np.ones(len(x)),x)))
 taus,theta=b._fit(coupon(tr,X[:n]),dt,t,target(tr)); print('material',np.round(taus,6),np.round(theta,6),'train error',qtile((b._design(coupon(tr,X[:n]),dt,t,taus)@theta-target(tr))[:,:-1]))
 full=np.r_[t,0.]; drifts={'linear':np.r_[t,1.],'log':np.r_[np.log1p(t),1.],'bounded':np.r_[-np.expm1(-t),1.]}
 both={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
 for key,x in [('train+diag',b._features(tr,g)[n:]),('all',b._features(both,g)[len(X):]),('diag',b._features(g,g)[:len(g['lab_id'])])]:
  past,hold,feat=coupon(g,x); dd=b._design((past,hold,feat),dt,t,taus); k=np.vstack((-np.expm1(-t[:,None]/taus),1/taus)); hs=hold[:,None,None]*k[None,:,:]; gap=dd[:,:,:2]; drift=feat@theta[2:]; Y=target(g)
  res=[]
  for mem in ['retained','reset','retain0','retain1','blendfit']:
   for dn,dr in drifts.items():
    mems={'retained':gap,'reset':hs,'retain0':np.stack((gap[:,:,0],hs[:,:,1]),axis=-1),'retain1':np.stack((hs[:,:,0],gap[:,:,1]),axis=-1)}
    if mem=='blendfit':
     delta=(gap-hs)*theta[:2]; yy=Y-(hs@theta[:2]+drift[:,None]*dr); mixture=np.linalg.lstsq(delta.reshape(-1,2),yy.ravel(),rcond=None)[0]; yp=hs@theta[:2]+delta@mixture+drift[:,None]*dr
    else: mixture=[]; yp=mems[mem]@theta[:2]+drift[:,None]*dr
    er=yp-Y; res.append((round(np.sqrt(np.mean(er[:,:-1]**2)),6),round(np.sqrt(np.mean(er[:,-1]**2)),6),mem,dn,np.round(mixture,3)))
  print('diag',key,'best',sorted(res,key=lambda z:z[:2])[:6])
  if key=='all':
   mats=np.stack((hs[:,:,0]*theta[0],hs[:,:,1]*theta[1],(gap-hs)[:,:,0]*theta[0],(gap-hs)[:,:,1]*theta[1]),axis=-1)
   for ix in [0,4,11,12]:
    M=np.column_stack((mats[:,ix,[0,2,3]],drift,np.ones(len(drift)))); beta=np.linalg.lstsq(M,Y[:,ix],rcond=None)[0]; print('diagnostic time',ix,'hold fast, retained fast/slow, drift, offset',np.round(beta,4),'resid',qtile(M@beta-Y[:,ix]))
