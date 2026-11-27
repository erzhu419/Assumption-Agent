exec(open('/tmp/rheology_drift_cv2.py').read().split('for kind,folds')[0])
from collections import defaultdict
out=defaultdict(list)
for mode,folds in [('coupon',[[i] for i in range(len(P['lab_id']))]),('lab',[np.flatnonzero(P['lab_id']==j) for j in labs])]:
 for ix in folds:
  q=np.ones(len(P['lab_id']),bool);q[ix]=0;a=sub(P,q);b=sub(P,ix)
  tau,w,y=m._spectrum(a,m._history(a,bridge),dt,t); pred=m._branches(m._history(b,bridge),dt,t,tau)@w; target=m._observations(b)
  for panelname,refs in [('TP',[T,P]),('RT',[R,T]),('T',[T])]:
   p=m._panel(refs)
   for fillname,frows in [('R',[R]),('RT',[R,T]),('TP',[T,P])]:
    f=m._fill(frows,p);x=m._design(a,p,f);z=m._design(b,p,f)
    for method in ('current','ols','ridge1','ridge3','scalar','scalar_offset'):
     if method=='current': beta=m._regress(x,y,.013);mu=z@beta
     elif method in ('ols','ridge1','ridge3'):
      lam={'ols':0,'ridge1':1,'ridge3':3}[method];dpen=np.array([0]+[lam]*4);beta=np.linalg.lstsq(np.r_[x,np.diag(dpen)],np.r_[y,dpen*beta0],rcond=None)[0];mu=z@beta
     else:
      A=np.c_[x[:,0],x[:,1:]@beta0[1:]];B=np.c_[z[:,0],z[:,1:]@beta0[1:]]
      if method=='scalar_offset':
       A=np.c_[A,a['lab_id'][:,None]==labs]; B=np.c_[B,b['lab_id'][:,None]==labs]
      pen=np.r_[[0,.15],np.repeat(.65,A.shape[1]-2)]
      coeff=np.linalg.lstsq(np.r_[A,np.diag(pen)],np.r_[y,pen*np.r_[[0,1],np.zeros(A.shape[1]-2)]],rcond=None)[0];mu=B@coeff
     error=pred+mu[:,None]*tt-target;out[mode,panelname,fillname,method].append((b['lab_id'],np.isfinite(b['coupon_features_raw']).all(1),error))
rms=lambda x:float(np.sqrt(np.mean(x*x)))
for mode in ('coupon','lab'):
 result=[]
 for k,vals in out.items():
  if k[0]!=mode:continue
  lab=np.concatenate([v[0] for v in vals]);c=np.concatenate([v[1] for v in vals]);e=np.concatenate([v[2] for v in vals]);sl=[lab==j for j in labs]+[c,~c];f=[rms(e[:,:-1]),rms(e[:,-1]),max(rms(e[q,:-1]) for q in sl),max(rms(e[q,-1]) for q in sl),rms(e[~c,:-1]),rms(e[~c,-1])];result.append((k[1:],np.round(f,5).tolist()))
 print(mode,'strain/rate overall, worst slice, missing');
 for k,f in sorted(result,key=lambda k:k[1][2])[:9]+[r for r in result if r[0] in [('TP','TP','current'),('TP','R','ols'),('TP','R','ridge3'),('TP','R','scalar')]]:print(k,f)
