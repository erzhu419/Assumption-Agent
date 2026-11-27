exec(open('/tmp/chamber_probe.py').read().split('for block,features')[0])
P,H,X=inp(diag,xd); T,Y=obs(diag); Q=np.c_[T,Y]; complete=np.isfinite(diag['coupon_features_raw']).all(1)
def expk(tau): return np.r_[-np.expm1(-t/tau),1/tau]
def intk(tau): return tau*expk(tau)
def model(p,kind):
 f,s=p[:2]; col=[expk(f),expk(s)]
 if kind=='linear': col += [np.r_[t,1.]]
 if kind=='third': col += [intk(p[2])]
 if kind=='repeat_fast': col += [np.r_[t*np.exp(-t/f),1.]]
 if kind=='repeat_slow': col += [np.r_[t*np.exp(-t/s),1.]]
 if kind=='quadratic': col += [np.r_[t**2,0.]]
 if kind=='linear_third': col += [np.r_[t,1.],intk(p[2])]
 return np.array(col).T
for kind in ('linear','third','repeat_fast','repeat_slow','quadratic','linear_third'):
 for freeze in (True,False):
  if freeze and 'third' not in kind: continue
  def decode(v): return np.r_[taus,v] if freeze else v
  def fun(v,ret=False):
   K=model(decode(v),kind); amp=np.linalg.lstsq(K,Q.T,rcond=None)[0].T; err=Q-amp@K.T
   return (amp,err) if ret else err.ravel()
  candidates=[]
  for tau3 in (.18,.4,1.,1.5,2.4,5.,12.):
   start=np.r_[taus,tau3] if 'third' in kind else taus
   fit=least_squares(fun,start[2:] if freeze else start,bounds=(.055,30),max_nfev=500)
   candidates.append(fit)
   if 'third' not in kind: break
  fit=min(candidates,key=lambda f:np.linalg.norm(f.fun)); pars=decode(fit.x); coef,e=fun(fit.x,True)
  print(kind,'fixed',freeze,'taus',pars.round(5),'curve/rate rmse',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)))
  if kind in ('linear','third','repeat_fast','repeat_slow') and np.sqrt(np.mean(e**2))<.0042:
   for desc,Z in [('all',np.c_[P,H,X]),('complete',np.c_[P,H,X][complete])]:
    C=coef if desc=='all' else coef[complete]; beta=np.linalg.lstsq(Z,C,rcond=None)[0]
    print(desc,'coeff rmse',np.sqrt(np.mean((C-Z@beta)**2,axis=0)).round(5),'beta\n',beta.T.round(4))
print('bridge max',np.max(np.abs(raw['calibration_load_cells']['bridge_raw'])), 'telemetry test min max',np.min(b['aux_chamber_telemetry'],axis=0),np.max(b['aux_chamber_telemetry'],axis=0))
