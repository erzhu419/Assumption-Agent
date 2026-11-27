exec(open('/tmp/explore98.py').read().split('for name,X in')[0])
X=align(g); spec=m._specimen(g,X,bridge); c,q=m._branches(spec,dt,t,tau); drift=w[2]+X@w[3:]; elastic=np.column_stack((c@w[:2],q@w[:2])); E=obs-elastic; E0=E[:,-1]; print('E0 vs drift',np.round(np.column_stack([g['lab_id'],drift,E0,E0-drift]),5));
for h,n in [(drift,'aligned'),(E0,'inferred')]:
 pred=[]; coeff=[]
 for j in range(len(t)):
  v=E[:,j]; p=least_squares(lambda b:(h*b[0]-v)/.002,[1.],loss='soft_l1').x[0]; coeff.append(p); pred.append(h*p-v)
 print(n,'kernel',np.round(coeff,6),'MAD per time',np.round(np.median(np.abs(pred),axis=1),6));print(n,'kernel/t tends',np.round(coeff/t,6))
U,S,V=np.linalg.svd(E[:,:-1],full_matrices=False); print('SVD',S.round(6));print('kernel norm ratios',(V[0]/V[0,0]).round(6));print('reconstruction row mad', np.median(abs(E[:,:-1]-(U[:,:1]*S[:1])@V[:1]),axis=1).round(6))
# Fit simple constitutive kernels, maintaining unit initial derivative.
for name,fn,start,bounds in [
 ('exp saturation',lambda x,p:-np.expm1(-p[0]*x)/p[0],[.7],([.001],[20.])),
 ('exp fade',lambda x,p:x*np.exp(-p[0]*x),[.3],([0],[20.])),
 ('rational fade',lambda x,p:x/(1+p[0]*x),[.7],([0],[20.])),
 ('rational square',lambda x,p:x/(1+p[0]*x*x),[.15],([0],[20.])),
 ('saturation plus secular',lambda x,p:p[1]*(-np.expm1(-p[0]*x)/p[0])+(1-p[1])*x,[.7,1.],([.001,-5],[20.,5])),
 ('two branch plus secular',lambda x,p:p[0]*(-np.expm1(-x/tau[0]))+p[1]*(-np.expm1(-x/tau[1]))+(1-p[0]/tau[0]-p[1]/tau[1])*x,[.3,2.],([-8,-8],[8,8])),
 ('log fade',lambda x,p:np.log1p(p[0]*x)/p[0],[.8],([.001],[30.]))]:
 pp=[]; coef=[]
 for i in range(len(r)):
  k=np.arange(len(r))!=i
  def res(p):return ((drift[k,None]*fn(t,p)-E[k,:-1])/.005).ravel()
  p=least_squares(res,start,bounds=bounds,loss='soft_l1').x; pp.append(elastic[i]+drift[i]*np.r_[fn(t,p),1.]);coef.append(p)
 print('simple',name,stats(pp),np.median(coef,axis=0).round(6))
