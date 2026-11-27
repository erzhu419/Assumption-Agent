import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; te=d['test']; ch=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']
train,test=old._prepare(d)
def obs(s): return np.column_stack([s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']])
y=obs(tr); tau,b=old._fit(train,dt,t,y); print('train tau,theta',tau,b)
def metrics(r):
 return np.round([[np.median(abs(a)),np.sqrt(np.mean(a*a)),np.quantile(abs(a),.9)] for a in [r[:,:-1],r[:,-1]]],7).tolist()
print('train fit',metrics(old._design(train,dt,t,tau)@b-y))
if ch is None: raise SystemExit
# Keep the public train/test panel in the feature estimate for the chamber probe.
merged={k:np.concatenate([te[k],ch[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}
x=old._features(tr,merged)[-len(ch['lab_id']):]; offset,total=old._calibration(d['calibration_load_cells']); past=(ch['past_channel_raw']-offset)@total; hold=(ch['hold_channel_raw']-offset)@total; drift=np.c_[np.ones(len(x)),x]; cu=(past,hold,drift); Y=obs(ch); A=old._design(cu,dt,t,tau)
print('baseline chamber',metrics(A@b-Y),'signed residual mean',np.round(np.mean(Y-A@b,axis=0),6)); print('chamber signal',metrics(Y))
S=np.r_[np.repeat(.008,len(t)),.02]
def assess(name,design,base=None):
 if base is None: base=np.zeros_like(Y)
 z=(design/S[None,:,None]).reshape(-1,design.shape[-1]); yy=((Y-base)/S).ravel(); coef=np.linalg.lstsq(z,yy,rcond=None)[0]; coef=least_squares(lambda q:z@q-yy,coef,loss='soft_l1').x
 print(name, 'coefs',np.round(coef,6), 'err',metrics(design@coef+base-Y))
assess('free theta',A)
assess('branch and drift gains',np.dstack([A[:,:,0]*b[0],A[:,:,1]*b[1],A[:,:,2:]@b[2:]]))
for name,hist,delta in [('rev stress',past[:,::-1],dt),('rev pairs',past[:,::-1],dt[::-1]),('zero',np.zeros_like(past),dt),('last',np.repeat(past[:,-1:],past.shape[1],axis=1),dt),('mean',np.repeat(past.mean(axis=1,keepdims=True),past.shape[1],axis=1),dt)]:
 Q=old._design((hist,hold,drift),delta,t,tau); print(name,'fixed',metrics(Q@b-Y)); assess(name+' free theta',Q)
# Separate hold and individual history contributions for both branches.
gram=np.column_stack((hold,past)); kernels=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); B=(gram[:,None,:,None]*kernels[None,:,None,:]).reshape(len(hold),len(t)+1,-1)
assess('free history (hold,step 0..5; fast,slow)',np.dstack([B,A[:,:,2:]]))
print('baseline linear coefficients in that layout:'); q=np.exp(-dt[:,None]/tau); print(np.vstack([b[:2],-(1-q)*np.vstack([np.cumprod(q[:0:-1],axis=0)[::-1],np.ones(2)])*b[:2]]));print('drift',b[2:]); print('cal affine raw RMSE',np.sqrt(np.mean((((d['calibration_load_cells']['bridge_raw']-offset)@total)-d['calibration_load_cells']['true_channel_loads'].sum(1))**2)))
