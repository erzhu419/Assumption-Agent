import contextlib,io,json
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/m99_probe.py').read())
p,h,D=cc; X=D[:,1:]; r=D@theta[2:]; Y=np.column_stack((ch['strain_curves_obs']-ch['strain_0'][:,None],ch['instant_rate_obs'])); labs=ch['lab_id']; H=np.r_[-np.expm1(-t[:,None]/tau)@theta[:2],theta[:2]@(1/tau)]; E=Y-h[:,None]*H
U,S,V=np.linalg.svd(E,full_matrices=False); print('hold constrained SVD',S.round(5)); K=V[0]/V[0,-1]; a=E@K/(K@K); print('times and empirical kernel',np.c_[np.r_[t,0],K].round(5)); print('closure max/rms',abs(E-a[:,None]*K).max(),np.std(E-a[:,None]*K)); print('rows: lab, stratum, hold, r, latent, r residual from linear fit, missing flags'); print(np.c_[labs,ch['stratum'],h,r,a,a-np.polyval(np.polyfit(r,a,1),r),np.isnan(ch['coupon_features_raw'])].round(5));
print('prior preprocessing:'); import inspect; print(inspect.getsource(m._features)); print(inspect.getsource(m._coupon)); print('chamber fields',list(ch)); print('dataset fields',list(d));
for lab in np.unique(labs):
 mask=labs==lab; print('lab',lab,'raw/std/drift',np.c_[ch['coupon_features_raw'][mask],X[mask],r[mask],a[mask]].round(4));
for name,M in [('features',D),('features plus squared drift',np.c_[D,r*r]),('features plus squares',np.c_[D,X*X]),('all pair products',np.c_[D,*[X[:,j]*X[:,k] for j in range(4) for k in range(j,4)]]),('features and missing',np.c_[D,np.isnan(ch['coupon_features_raw'])]),('features and hold',np.c_[D,h]),('features and auxiliary',np.c_[D,ch['aux_chamber_telemetry']] if 'aux_chamber_telemetry' in ch else D)]:
 pred=np.empty_like(Y)
 for lab in np.unique(labs):
  q=labs!=lab; B=np.linalg.lstsq(M[q],E[q],rcond=None)[0]; pred[~q]=M[~q]@B
 print(name,'train latent rms',np.std(a-M@np.linalg.lstsq(M,a,rcond=None)[0]),'heldout curve/rate max',np.max(abs(pred[:,:-1]-E[:,:-1])),np.max(abs(pred[:,-1]-E[:,-1])))
from scipy.optimize import least_squares
for name,fn,start in [('log',lambda g:np.log1p(g*t)/g,.5),('rational',lambda g:t/(1+g*t),.3),('exp',lambda g:-np.expm1(-g*t)/g,.4),('sqrt rational',lambda g:t/np.sqrt(1+g*t),.8),('rational squared time',lambda g:t/(1+g*t*t),.1)]:
 fit=least_squares(lambda u:fn(u[0])-K[:-1],[start],bounds=(.00001,20)); print(name,fit.x,np.max(abs(fit.fun)))
idx=json.load(open('/holdout_task/public_asset_index.json')); print('asset index top-level',list(idx)); print('asset content hints',[(k,str(v)[:260]) for k,v in idx.items()]);
