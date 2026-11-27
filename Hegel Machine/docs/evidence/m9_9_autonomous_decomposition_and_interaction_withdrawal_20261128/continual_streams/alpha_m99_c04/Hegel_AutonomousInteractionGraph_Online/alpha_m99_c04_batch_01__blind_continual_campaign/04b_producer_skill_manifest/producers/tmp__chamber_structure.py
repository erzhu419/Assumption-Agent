import pickle,runpy,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=160)
p=dict(np.load('/tmp/chamber_probe.npz')); globals().update(p)
a=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c04_s00_solver.py'); i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; cal=ds['calibration_load_cells']; n,k=y.shape
R=np.c_[np.ones(len(cal['bridge_raw'])),cal['bridge_raw']]; print('linear bridge residual by channel',np.sqrt(np.mean((R@np.linalg.lstsq(R,cal['true_channel_loads'],rcond=None)[0]-cal['true_channel_loads'])**2,axis=0)))
for g in [tr,te,d]:
 print('reference stats',[(l,len(g['lab_id'][g['lab_id']==l]),round(np.median(g['ref_coupons'][g['lab_id']==l].mean(0)),3),round(np.median(g['ref_coupons'][g['lab_id']==l].std(0)),3)) for l in np.unique(g['lab_id'])], 'missing',np.isnan(g['coupon_features_raw']).sum(0))
age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; mem=lambda q: past@(-np.expm1(-dt[:,None]/q)*np.exp(-age[:,None]/q))
z=mem(tau); gap=hold[:,None]-z
w=np.r_[np.full(k-1,1/max(sc[0],1e-6)),1/max(sc[1],1e-6)]; idx=np.arange(n)
def fit(des, ii=idx, free=True):
 def design(q,ii):
  B=np.vstack([np.c_[-np.expm1(-t[:,None]/q),t],np.r_[1/q,1]])
  return np.column_stack([(D[ii,None,:]*B[None,:,j,None]).reshape(len(ii)*k,-1) for j,D in enumerate(des)])
 yw=(y[ii]*w).ravel(); ww=np.tile(w,len(ii))
 def prof(lq, returnbeta=False):
  A=design(np.exp(lq),ii)*ww[:,None]; beta=np.linalg.lstsq(A,yw,rcond=None)[0]; return beta if returnbeta else A@beta-yw
 lq=np.log(tau)
 if free:
  opt=least_squares(prof,lq,bounds=(np.log([.25,1.25]),np.log([1.15,12.])),diff_step=1e-4); lq=opt.x
 return np.exp(lq),np.split(prof(lq,True),np.cumsum([D.shape[1] for D in des])[:-1]),(design(np.exp(lq),idx)@prof(lq,True)).reshape(n,k)
O=np.ones(n); H=np.c_[O,hold]; Hd=np.c_[H,dr]; Hg=np.c_[H,z]; Hgd=np.c_[Hg,dr]; Hp=np.c_[H,past]; Hpd=np.c_[Hp,dr]; Hpx=np.c_[Hp,x]; dx=np.c_[O,x]
models={'hold+dr':[Hd]*3,'hold+mem+dr':[Hgd]*3,'all histories':[Hpd]*3,'all features':[Hpx]*3,'causal sparse':[np.c_[H,z[:,0]],Hgd,np.c_[O,dr]],'two gaps+dr':[np.c_[O,gap,dr]]*3,'instant-hold fast':[H,Hgd,np.c_[O,dr]],'fast and slow linear drift':[Hgd,Hgd,dx]}
for name,des in models.items():
 for free in (False,True):
  q,c,yhat=fit(des,free=free); e=yhat-y; cv=np.zeros_like(y)
  for l in np.unique(d['lab_id']):
   sel=d['lab_id']==l; cv[sel]=fit(des,idx[~sel],free)[2][sel]-y[sel]
  print(name,free,'poles',q,'train rmse',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),'LOLO',np.round([np.sqrt(np.mean(cv[:,:-1]**2)),np.sqrt(np.mean(cv[:,-1]**2)),max(np.sqrt(np.mean(cv[d['lab_id']==l,:-1]**2)) for l in np.unique(d['lab_id']))],5),'coef',c)
