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
print('component true/explicit sweep and correction, by coupon:'); print(np.round(np.c_[lab,st,h,intrinsic,C,R,(H[:,None]-Z)*th[:2]],4));
for name,D in [('base',np.c_[f,h,z]),('true sweep',np.c_[f,H,Z]),('all stress',np.c_[f,h,z,H,Z]),('all past',np.c_[f,h,past]),('lab only',np.c_[f,h,z,np.eye(4)[lab,1:]]),('stratum',np.c_[f,h,z,np.eye(3)[st,1:]]),('lab and stratum',np.c_[f,h,z,np.eye(4)[lab,1:],np.eye(3)[st,1:]]),('stress cal vars',np.c_[f,h,z,h*g,z*g[:,None],offset]),('cmd)',np.c_[f,(h-offset)/g,((past-offset[:,None])/g[:,None])@K])]:
 pred=np.empty_like(R); predl=np.empty_like(R); P=np.linalg.lstsq(D,R,rcond=None)[0];
 for i in range(n):
  sel=np.arange(n)!=i; pred[i]=D[i]@np.linalg.lstsq(D[sel],R[sel],rcond=None)[0]
 for l in np.unique(lab):
  sel=lab!=l; predl[~sel]=D[~sel]@np.linalg.lstsq(D[sel],R[sel],rcond=None)[0]
 err=lambda pred:np.round(np.sqrt(np.mean(((pred-R)@B.T)**2,axis=0))[[0,-2,-1]],4)
 print(name,'fit/loco/lolo',err(D@P),err(pred),err(predl),'coefs',np.round(P,3).tolist() if name in ('base','stratum','true sweep') else '')
