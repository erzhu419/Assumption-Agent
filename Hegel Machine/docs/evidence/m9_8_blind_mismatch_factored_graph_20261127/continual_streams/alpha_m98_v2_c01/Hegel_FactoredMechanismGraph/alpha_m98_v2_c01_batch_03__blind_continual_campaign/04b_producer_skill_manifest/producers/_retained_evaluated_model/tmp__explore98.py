import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c01_batch_02_solver as m
import alpha_m98_v2_c01_batch_01_solver as old
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=x['dataset'];tr,te,g=d['train'],d['test'],x['campaign_diagnostics']; t,dt=d['eval_times'],d['past_dt'];cal=d['calibration_load_cells'];L,R=cal['true_channel_loads'],cal['bridge_raw'];bridge=m._axial_load_map(cal)
print('cal keys',[(k,type(v).__name__) for k,v in cal.items()]); print('sweep shape',[(k,np.shape(v)) for k,v in cal.get('single_channel_sweeps',{}).items()] if isinstance(cal.get('single_channel_sweeps'),dict) else str(type(cal.get('single_channel_sweeps')))); print('cal range',L.min(0),L.max(0));print('labs',[(np.unique(s['lab_id'],return_counts=True),np.unique(s['stratum'],return_counts=True)) for s in (tr,te,g)])
from itertools import combinations_with_replacement
for order in (1,2,3):
 A=np.column_stack([np.ones(len(L))]+[np.prod(L[:,ks],axis=1) for o in range(1,order+1) for ks in combinations_with_replacement(range(4),o)]);E=[]
 for i in range(5):
  test=np.arange(len(L))%5==i; E.extend(A[test]@np.linalg.lstsq(A[~test],R[~test],rcond=None)[0]-R[test]);
 print('cal forward CV',order,np.sqrt(np.mean(np.square(E),axis=0)))
xt,xe=m._features(tr,te); tau,w=m._fit(m._specimen(tr,xt,bridge),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);print('tau,w',tau,w)
y=g['strain_curves_obs']-g['strain_0'][:,None];r=g['instant_rate_obs']; obs=np.column_stack((y,r));print('diag obs typical',np.median(np.abs(obs),axis=0))
def align(g):
 keys=('lab_id','stratum','ref_coupons','coupon_features_raw'); joint={k:np.concatenate([tr[k],te[k]]) for k in keys}; X=g['coupon_features_raw'].copy(); J=joint['coupon_features_raw'].copy();
 for lab in np.unique(g['lab_id']):
  rows=joint['lab_id']==lab; ref=joint['ref_coupons'][rows]; center=np.median(ref.mean(0));scale=np.median(ref.std(0)); X[g['lab_id']==lab]=(X[g['lab_id']==lab]-center)/scale;J[rows]=(J[rows]-center)/scale
 _,idx,cnt=np.unique(np.column_stack([joint[k] for k in keys[:2]]),axis=0,return_inverse=True,return_counts=True); W=1/cnt[idx,None]; mean=np.nansum(J*W,axis=0)/np.sum(W*np.isfinite(J),axis=0);return np.where(np.isfinite(X),X,mean)
def stats(p):
 e=p-obs;return np.round([np.median(abs(e[:,:-1])),np.sqrt(np.mean(e[:,:-1]**2)),np.max(np.median(abs(e[:,:-1]),axis=1)),np.median(abs(e[:,-1])),np.sqrt(np.mean(e[:,-1]**2)),max(abs(e[:,-1]))],7)
def profile(F):
 ps=[];co=[]
 for i in range(len(r)):
  keep=np.arange(len(r))!=i; A=F[keep].reshape(-1,F.shape[-1]); B=obs[keep].ravel(); p=np.linalg.lstsq(A,B,rcond=None)[0];
  sc=np.tile(np.r_[np.repeat(.0003,len(t)),.0006],sum(keep));p=least_squares(lambda p:(A@p-B)/sc,p,loss='soft_l1').x;ps.append(F[i]@p);co.append(p)
 print('   CV',stats(ps),'coef',np.round(np.median(co,axis=0),6))
for name,X in [('train_test_panel',align(g)),('train_diag_panel',m._features(tr,g)[1])]:
 spec=m._specimen(g,X,bridge);past,h,_=spec; drift=w[2]+X@w[3:]; c,q=m._branches(spec,dt,t,tau); kc=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); base=np.column_stack((c@w[:2]+drift[:,None]*t,q@w[:2]+drift));reset,rr=old._chamber_response(h,drift,g['strain_0'],t,tau,w);print(name,'base',stats(base),'oldreset',stats(np.column_stack((reset-g['strain_0'][:,None],rr))));print('drift',drift.round(5));print('hold',h.round(4));
 def outer(v,k):return v[:,None,None]*k[None,:,:]
 H=outer(h,kc); D=outer(drift,kc); P=H-np.concatenate((c,q[:,None,:]),axis=1); T=outer(drift,np.r_[t,1][:,None]);
 for label, f in [('H+D',(H,D)),('H+D+P',(H,D,P)),('H+D+P+T',(H,D,P,T)),('H+D+P+T+X',(H,D,P,T,*(outer(v,kc) for v in spec[2].T)))]:print(label);profile(np.concatenate(f,axis=-1))
