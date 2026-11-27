from probe import *
from scipy.optimize import least_squares
print('times',t,'dt',dt,'diagnostic n',len(R))
B, I = curve[:,:,:2] @ theta[:2], rate[:,:2] @ theta[:2]
for name,f,df in [('linear',t,1),('log',np.log1p(t),1),('sqrt',2*(np.sqrt(1+t)-1),1),('rational',t/(1+t),1),('saturating',-np.expm1(-t),1)]:
 e=Y-B-drift[:,None]*f; er=R-I-drift*df
 print(name,'horizon RMSE',np.sqrt(np.mean(e*e,axis=0)).round(5),'rate RMSE',np.sqrt(np.mean(er*er)))
print('train median, p95 curve/rate errors',*[np.quantile(abs(e),[.5,.95,.99,1]) for e in [y-prior._design(taus,tr,dt,t)[0]@theta,r-prior._design(taus,tr,dt,t)[1]@theta]])
print('diag rate residual',np.c_[drift,R-I,R-I-drift].round(4)); print('diag residual singulars',np.linalg.svd(Y-B,compute_uv=False).round(4))
print('time effective drift integral',(((Y-B).T@drift)/(drift@drift)).round(5))

def predict(v,model):
 a=np.exp(v[0]); dscale=v[1] if len(v)>1 else 1.
 if model=='log': f=np.log1p(a*t)/a
 elif model=='sqrt': f=2*(np.sqrt(1+a*t)-1)/a
 elif model=='rational': f=t/(1+a*t)
 else: f=-np.expm1(-a*t)/a
 return B+dscale*drift[:,None]*f,I+dscale*drift

def err(v,model,mask):
 p,q=predict(v,model); return np.r_[(p[mask]-Y[mask]).ravel(),(q[mask]-R[mask])*2]
for model in ['log','sqrt','rational','exp']:
 for k in [1,2]:
  full=least_squares(err,[0.,1.][:k],args=(model,np.ones(len(R),bool)),loss='soft_l1',f_scale=.008).x
  es=[]
  for lab in np.unique(cd['lab_id']):
   m=cd['lab_id']==lab; v=least_squares(err,[0.,1.][:k],args=(model,~m),loss='soft_l1',f_scale=.008).x
   p,q=predict(v,model); e=p[m]-Y[m]; es.append([np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e[:,-1]**2)),np.sqrt(np.mean((q[m]-R[m])**2))]);
  print('fit',model,k,'a',np.exp(full[0]),'scale',full[1:],'worst LOLO curve/final/rate RMSE',np.max(es,axis=0).round(6))
