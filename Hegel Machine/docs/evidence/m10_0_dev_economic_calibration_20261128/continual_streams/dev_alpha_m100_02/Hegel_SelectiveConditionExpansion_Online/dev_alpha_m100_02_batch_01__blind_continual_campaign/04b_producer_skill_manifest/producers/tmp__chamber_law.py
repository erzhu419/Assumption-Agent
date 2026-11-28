import pickle, numpy as np, sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_alpha_m100_02_s00_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; q=p['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; cal=m._bridge(d['calibration_load_cells']); out=m.solve_batch(d); tau,th=out['estimated_taus'],out['hereditary_weights']; lab=q['lab_id']; st=q['stratum']; n=len(lab)
print('tau',tau,'theta',th,'dt',dt,'t',t)
panel={k:np.concatenate((tr[k],te[k]),axis=0) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
x=m._features(panel,q)[1]; print('diagnostic raw shape and missing',x.shape,np.isnan(q['coupon_features_raw']).sum()); print('ref pooled lab params',[(int(l),np.median(panel['ref_coupons'][panel['lab_id']==l].mean(0)),np.median(panel['ref_coupons'][panel['lab_id']==l].std(0))) for l in np.unique(lab)])
past,h,f=m._prepare(q,x,cal); print('past/hold range',past.min(),past.max(),h.min(),h.max())
y=np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs'])); a,b=m._design(np.log(tau),(past,h,f),dt,t); base=np.column_stack((a@th,b@th)); print('baseline median/RMSE/max abs',np.median(abs(y-base),axis=0)[[0,-2,-1]],np.sqrt(np.mean((y-base)**2,axis=0))[[0,-2,-1]],np.max(abs(y-base),axis=0)[[0,-2,-1]])
def basis(u):
 v=np.exp(u); return np.vstack((np.c_[-np.expm1(-t[:,None]/v),t],np.r_[1/v,1]))
def project(u):
 B=basis(u); return np.linalg.lstsq(B,y.T,rcond=None)[0].T
u=np.log(tau); B=basis(u); C=project(u); opt=least_squares(lambda uu:(project(uu)@basis(uu).T-y).ravel(),u,loss='soft_l1',f_scale=.003); print('temporal base/opt RMSE, opt tau:',np.sqrt(np.mean((C@B.T-y)**2)),np.sqrt(np.mean((project(opt.x)@basis(opt.x).T-y)**2)),np.exp(opt.x)); print('noise all slices',np.round(np.sqrt(np.mean((C@B.T-y)**2,axis=0)),5))
sw=d['calibration_load_cells']['single_channel_sweeps']; pars=np.array([np.polyfit(sw['cmd_stress'][sw['lab_id']==l],sw['true_stress'][sw['lab_id']==l],1) for l in range(4)]); g,offset=pars[lab].T; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0]; K=np.exp(-tail[:,None]/tau)*-np.expm1(-dt[:,None]/tau); z=past@K; Z=(g[:,None]*past+offset[:,None])@K; H=g*h+offset; intrinsic=f@th[2:]; A=np.c_[(h[:,None]-z)*th[:2],intrinsic]; R=C-A
from itertools import product
from scipy.linalg import lstsq
stress=h[:,None]*th[:2]; v=intrinsic; base_parts=np.c_[stress,v]; print('hold-only coefficients residual corr:'); print(np.corrcoef(np.c_[C-base_parts,v,z].T).round(3)); print('component OLS',np.linalg.lstsq(np.c_[np.ones(n),v,h,z],C,rcond=None)[0].round(5)); print('raw fully projected component OLS',np.linalg.lstsq(np.c_[np.ones(n),v,h,z],y,rcond=None)[0][:,[0,4,11,12]].round(5)); print('drift regression against drift',np.linalg.lstsq(np.c_[np.ones(n),v],C[:,2],rcond=None)[0]); print('diag vs test ranges');
for part,xx in [(q,x),(te,xt if 'xt' in globals() else m._features(tr,te)[1])]:
 pp,hh,ff=m._prepare(part,xx,cal); print(np.quantile(hh,[0,.5,1]),np.quantile(ff@th[2:],[0,.5,1]))
W=np.r_[np.ones(len(t)),1.];
def fit(cols, fixed, labels):
 M=np.stack(cols,axis=-1); dy=y-fixed; errs=[]; scores=[]
 for name,folds in [('fit',[np.ones(n,dtype=bool)]),('coupon',[np.arange(n)==i for i in range(n)]),('lab',[lab==l for l in np.unique(lab)]),('stratum',[st==s for s in np.unique(st)])]:
  ep=[]; worst=[]
  for test in folds:
   train=~test if name!='fit' else test; coef=lstsq((M[train]*W[None,:,None]).reshape(-1,M.shape[-1]),(dy[train]*W).ravel())[0]; err=(M[test]@coef-dy[test]); ep.extend(err); worst.append((np.sqrt(np.mean(err[:,:-1]**2)),np.sqrt(np.mean(err[:,-1]**2))))
  ep=np.array(ep); scores.append((np.sqrt(np.mean(ep[:,:-1]**2)),np.sqrt(np.mean(ep[:,-1]**2)),*np.max(worst,axis=0)))
  if name=='fit': print(labels,'coefs',coef.round(5))
 print(np.round(scores,5))
def term(x,b): return np.asarray(x)[:,None]*B[:,b]
for fixedname,fixed in [('hold',stress@B[:,:2].T),('gap',(stress-z*th[:2])@B[:,:2].T)]:
 for label,extra in [('none',[]),('memory',[term(z[:,i],i) for i in range(2)]),('hold',[term(h,i) for i in range(2)]),('freeX',[term(f[:,j],i) for j in range(1,5) for i in range(3)])]:
  cols=[term(v,i) for i in range(3)]+[term(np.ones(n),i) for i in range(3)]+extra; fit(cols,fixed,fixedname+' '+label)
print('drift regression encoded slopes columns curves+rate:'); print(np.linalg.lstsq(np.c_[np.ones(n),v,h,z],y-stress@B[:,:2].T,rcond=None)[0].round(4))
