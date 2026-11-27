exec(open('/tmp/chamber_amplitude.py').read().split('zsets=')[0].replace("print(np.c_[C,H,G,D,U].round(4))", "pass"))
_,qtest=m._features(b,c); _,qself=m._features(c,c)
for label, xx in [('train+test',Q),('train+diag',q),('test+diag',qtest),('diag-only',qself)]:
 dd=theta[2]+xx@theta[3:]; X=np.c_[H,dd]
 E=C-X@np.linalg.lstsq(X,C,rcond=None)[0]
 print(label,'feature ranges',np.round(xx.min(0),2),np.round(xx.max(0),2),'hold + drift projection',np.sqrt(np.mean(E**2,axis=0)).round(4),'rate residual by lab',[(l,round(np.sqrt(np.mean((E[lab==l]@F[-1])**2)),4)) for l in np.unique(lab)])
 print('fast implied drift deviations',np.c_[lab,(C[:,0]-theta[0]*H)/.18-dd].round(4).tolist())
# Use coupon-level held-out labs, measuring both initial rate and complete curves.
obs=np.column_stack((Y,R)); basis=F.T; Z0,Z1=Z.T
for label,xx in [('train+test',Q),('diag-only',qself)]:
 dd=theta[2]+xx@theta[3:]; base=np.c_[theta[0]*H,theta[1]*H,np.zeros(len(H))]@basis
 hist=[-Z0,-Z1]; terms=[('d0',dd,0),('d1',dd,1),('d2',dd,2)]
 extras=[('z0',-Z0,0),('z1',-Z1,1),('h1',H,1),('gh1',H*Z1,1),('dabs1',dd*abs(dd),1),('d^3_1',dd**3,1),('hd1',H*dd,1),('z1d1',Z1*dd,1),('z0d1',Z0*dd,1),('z1^2',Z1**2,1),('z1sat',Z1/(1+Z1**2),1)]
 def assess(ts, report=True):
  Dm=np.stack([v[:,None]*basis[j] for _,v,j in ts],axis=-1); target=obs-base
  def fit(s): return np.linalg.lstsq(Dm[s].reshape(-1,len(ts)),target[s].ravel(),rcond=None)[0]
  E=np.empty_like(obs)
  for l in np.unique(lab):
   s=lab!=l; E[~s]=Dm[~s]@fit(s)-target[~s]
  coef=fit(np.ones(len(H),bool)); e=Dm@coef-target
  rms=lambda e: np.sqrt(np.mean(e**2))
  vals=[rms(e),rms(E[:,:-1]),rms(E[:,-1]),max(rms(E[lab==l]) for l in np.unique(lab))]
  if report: print(label,[t[0] for t in ts],np.round(vals,5),coef.round(5))
  return vals
 assess(terms); assess(terms+extras[:2]); assess(terms+[extras[1], extras[2]])
 for term in extras[3:]: assess(terms+[extras[1],term])
 # Check if unexplained signal is restricted to rows with imputed features.
 Dm=np.stack([v[:,None]*basis[j] for _,v,j in terms],axis=-1); est=np.linalg.lstsq(Dm.reshape(-1,len(terms)),(obs-base).ravel(),rcond=None)[0]; e=obs-base-Dm@est
 print('row residual RMS and missing:',np.c_[lab,np.isnan(c['coupon_features_raw']).sum(1),np.sqrt(np.mean(e**2,axis=1)),e[:,-1]].round(4).tolist())
print('telemetry mean/std',[(k,np.mean(d[k]['aux_chamber_telemetry'],axis=0).round(4),np.std(d[k]['aux_chamber_telemetry'],axis=0).round(4)) for k in ('train','test')])
