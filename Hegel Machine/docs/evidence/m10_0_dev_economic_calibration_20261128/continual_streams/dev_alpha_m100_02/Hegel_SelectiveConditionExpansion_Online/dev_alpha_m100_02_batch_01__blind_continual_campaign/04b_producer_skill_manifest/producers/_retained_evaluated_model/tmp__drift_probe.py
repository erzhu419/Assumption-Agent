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
v=intrinsic; vq=m._features(q,q)[0]@th[3:]+th[2]; vt=m._features(tr,q)[1]@th[3:]+th[2]; missing=np.isnan(q['coupon_features_raw']); Y=y-(h[:,None]*th[:2])@B[:,:2].T
u,s,w=np.linalg.svd(Y,full_matrices=False); inferred=Y@w[0]; inferred*=np.sign(np.corrcoef(inferred,v)[0,1]); print('hold residual singular values',s.round(5)); print('hold residual rank1 by coupon lab,missing,v,diag-only,train+diag,inferred:'); print(np.c_[lab,missing.sum(1),v,vq,vt,inferred].round(5));
for label,vv in [('pooled',v),('diag',vq),('train+diag',vt)]:
 for form,D in [('linear',np.c_[np.ones(n),vv]),('quadratic',np.c_[np.ones(n),vv,vv**2]),('cubic',np.c_[np.ones(n),vv,vv**2,vv**3]),('unscaled',vv[:,None]),('missing',np.c_[np.ones(n),vv,missing])]:
  e=Y-D@np.linalg.lstsq(D,Y,rcond=None)[0]; print(label,form,'RMSE',np.sqrt(np.mean(e**2,axis=0))[[0,11,12]],'worst rows',np.argsort(np.linalg.norm(e,axis=1))[-6:]);
print('drift basis coefs for rank1',np.linalg.lstsq(B,w[0],rcond=None)[0]); print('raw feature missing rows',np.flatnonzero(missing.any(1))); print('full observations fit (pooled)'); i=~missing.any(1); co=np.linalg.lstsq(np.c_[np.ones(n),v][i],Y[i],rcond=None)[0]; print(np.linalg.lstsq(B,co.T,rcond=None)[0].T.round(5)); print('errors',np.sqrt(np.mean((Y-np.c_[np.ones(n),v]@co)**2,axis=1)).round(4))
