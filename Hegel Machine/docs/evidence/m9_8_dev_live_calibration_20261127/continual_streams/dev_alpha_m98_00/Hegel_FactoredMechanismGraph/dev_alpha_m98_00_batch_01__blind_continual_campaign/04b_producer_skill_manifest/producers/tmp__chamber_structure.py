from rheology_base import *
P,H,X=pq; D=X@b[2:]; Y=np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs'])); n=len(H)
T=np.vstack((np.column_stack((-np.expm1(-t[:,None]/tau),t)),np.r_[1/tau,1])); C=np.linalg.lstsq(T,Y.T,rcond=None)[0].T
Xq=a._aligned_features(q,q); print('q only alignment shape',np.shape(Xq)); Xq=np.asarray(Xq[0] if isinstance(Xq,tuple) else Xq[:n]); Dq=Xq@b[2:]; print('alignment comparisons [lab,stratum,missing,D,Dq,A0-theta0*H,A1,drift]',np.round(np.c_[q['lab_id'],q['stratum'],np.isnan(q['coupon_features_raw']).sum(axis=1),D,Dq,C[:,0]-b[0]*H,C[:,1:]],4))
base=np.c_[np.ones(n),H,D]; sol=np.linalg.lstsq(base,Y,rcond=None)[0]; E=Y-base@sol; u,s,v=np.linalg.svd(E,full_matrices=False); print('residual singular values',np.round(s,4)); print('dominant residual time signatures',np.round(v[:3],3)); print('dominant signature coefficients',np.round(np.linalg.lstsq(T,v[:3].T,rcond=None)[0],3))
z=np.zeros((n,2)); zp=z.copy(); za=z.copy(); zs=z.copy()
for k,step in enumerate(dt):
 w=-np.expm1(-step/tau); p=P[:,k,None]; z+=w*(p-z); zp+=w*(p*p-zp); za+=w*(abs(p)-za); zs+=w*(np.tanh(p)-zs)
def audit(label,F):
 Q=np.linalg.qr(F)[0]; e=(Y-Q@(Q.T@Y))/(1-np.sum(Q*Q,axis=1))[:,None]; return label,*np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.max(np.sqrt(np.mean(e[:,:-1]**2,axis=1))),np.max(np.abs(e[:,-1]))],5)
print(audit('base',base)); dct={'Dq':Dq,'D2':D**2,'H2':H**2,'HD':H*D,'Dtanh':np.tanh(2*D),'Drational':D/(1+D**2)}
for j in range(2):
 for k,vv in [('z',z[:,j]),('z2',zp[:,j]),('za',za[:,j]),('zs',zs[:,j]),('g2',(H-z[:,j])**2),('zg',H*z[:,j]),('zD',D*z[:,j]),('zrat',z[:,j]/(1+z[:,j]**2))]:dct[k+str(j)]=vv
for j,p in enumerate(P.T):
 for key,vec in [('p',p),('p2',p*p),('pd',p*D),('ph',p*H),('pt',np.tanh(p)),('pr',p/(1+p*p))]: dct[key+str(j)]=vec
for i in range(P.shape[1]):
 for j in range(i+1,P.shape[1]):
  for key,vv in [('diff2',(P[:,i]-P[:,j])**2),('sum2',(P[:,i]+P[:,j])**2),('product',P[:,i]*P[:,j])]:dct[f'{key}_{i}{j}']=vv
for j,x in enumerate(X[:,1:].T):dct[f'x{j}']=x;dct[f'x2_{j}']=x*x
for j in range(3):dct[f'lab{j}']=(q['lab_id']==j+1).astype(float);dct[f'stratum{j}']=(q['stratum']==j).astype(float)
print('single additions',sorted((audit(k,np.c_[base,v]) for k,v in dct.items()),key=lambda r:r[1]+r[2])[:18]);
for label,F in [('hold nonlinear',np.c_[base,H**2]),('features',np.c_[np.ones(n),H,X[:,1:]]),('diagnostic alignment',np.c_[np.ones(n),H,Dq])]:print(audit(label,F));
for lo in np.unique(q['lab_id']):
 idx=q['lab_id']!=lo; cf=np.linalg.lstsq(base[idx],Y[idx],rcond=None)[0]; print('base held-out lab',lo,np.sqrt(np.mean((Y[~idx]-base[~idx]@cf)**2,axis=0))[[0,5,11,12]].round(4))
