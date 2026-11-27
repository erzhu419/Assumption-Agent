import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; dt=D['past_dt']; t=D['eval_times']; a=C['bridge_raw']; y=C['true_channel_loads']; b=np.linalg.lstsq(np.c_[np.ones(len(a)),a],y,rcond=None)[0]; print('bridge',b.round(5),'rms',np.mean((np.c_[np.ones(len(a)),a]@b-y)**2)**.5)
def inputs(G):
 lab=G['lab_id']; ref=G['ref_coupons']; x=G['coupon_features_raw'].copy();
 for l in np.unique(lab):
  q=lab==l; x[q]=(x[q]-np.median(ref[q].mean(0)))/np.median(ref[q].std(0))
 _,ix,n=np.unique(np.c_[lab,G['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[ix]; x=np.where(np.isnan(x),np.nansum(x*w[:,None],axis=0)/np.sum(np.isfinite(x)*w[:,None],axis=0),x)
 return G['past_channel_raw']@b[1:].sum(1)+b[0].sum(),G['hold_channel_raw']@b[1:].sum(1)+b[0].sum(),x
def basis(g,taus,mode=0,aux=False):
 p,h,x=inputs(g); q=np.exp(-dt[:,None]/taus); state=np.zeros((len(x),2));
 for j in range(len(dt)): state=q[j]*state+(1-q[j])*p[:,j,None]
 z=h[:,None]-state
 if mode==1: z*=taus
 if mode==2: z=h[:,None]*taus-state
 if mode==3: z=h[:,None]
 v=np.c_[np.ones(len(x)),x];
 if aux: v=np.c_[v,g['aux_chamber_telemetry']]
 F=np.concatenate((z[:,None,:]*(-np.expm1(-t[:,None]/taus)),v[:,None,:]*t[None,:,None]),axis=2)
 R=np.c_[z/taus,v]; return np.concatenate([F,R[:,None,:]],axis=1)
def target(g): return np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
def fit(g,mode=0,aux=False):
 yy=target(g); p=len(basis(g,np.array([.5,3]),mode,aux)[0,0]); rms=lambda e: np.sqrt(np.mean(e**2));
 def err(v): return basis(g,np.exp(v[:2]),mode,aux)@v[2:]-yy
 s=np.r_[np.full(len(t),.025),.045]; best=None
 for tau in [[.23,1.3],[.6,3],[1.5,7]]:
  F=basis(g,np.array(tau),mode,aux); th=np.linalg.lstsq(F.reshape(-1,p),yy.ravel(),rcond=None)[0]; v=least_squares(lambda v:(err(v)/s).ravel(),np.r_[np.log(tau),th],bounds=(np.r_[[np.log(.05001)]*2,[-np.inf]*p],np.r_[[np.log(24.999)]*2,[np.inf]*p]),loss='soft_l1',max_nfev=150)
  if best is None or v.cost<best.cost: best=v
 v=best.x; e=err(v); order=np.argsort(v[:2]); v[:2]=v[:2][order];v[2:4]=v[2:4][order]; print('fit',mode,aux,'cost',round(best.cost,2),'taus',np.exp(v[:2]).round(4),'theta',v[2:].round(5),'curve/rate MAE',np.mean(abs(e[:,:-1])).round(5),np.mean(abs(e[:,-1])).round(5),'median by time',np.median(abs(e),0).round(4)); return v
if __name__=='__main__':
 for k in ['train','test','chamber_verification_trials']:
  g=D[k]; print(k, None if g is None else [(j,np.shape(a)) for j,a in g.items()]);
  if g is None: continue
  print('counts',np.unique(np.c_[g['lab_id'],g['stratum']],axis=0,return_counts=True)); print('aux',np.asarray(g.get('aux_chamber_telemetry',[]))[:5]);
  if 'strain_curves_obs' in g:
   for mode in [0,2,3]: fit(g,mode)
   if 'aux_chamber_telemetry' in g: fit(g,0,True)
