exec(open('/tmp/scalar.py').read().split('for name,u')[0]); from scipy.special import erf
F=a['_readout'](w)
links={'tanh': np.tanh,'rational':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'erf':erf,'atan':np.arctan}
forms=[(h,b) for h in links for b in ['both','inner','none']]
idx=np.arange(len(y)); s=np.abs(F@fit(F)); norms=np.max(abs(w),axis=1)
splits=[(idx[:52],idx[52:]),(idx[20:],idx[:20]),(np.argsort(s)[:-12],np.argsort(s)[-12:]),(np.argsort(norms)[:-12],np.argsort(norms)[-12:])]+[(np.setdiff1d(idx,e),e) for e in np.array_split(np.random.default_rng(11).permutation(len(y)),4)]
for h,b in forms:
 f=links[h]; FF=F if b!='none' else F[:,1:]; res=[]
 for ii,jj in splits:
  def pred(p):
   q=FF@p[:-3]; return q+p[-3]*f(np.exp(p[-2])*q)+(p[-1] if b=='both' else 0)
  c=np.linalg.lstsq(FF[ii],y[ii],rcond=None)[0]; p=np.r_[.8*c,.4,.3,0]; m=least_squares(lambda p:(pred(p)-y)[ii],p,max_nfev=200); r=abs(pred(m.x)-y);res.append([r[ii].mean(),r[jj].mean(),r[jj].max()]);
  if ii[0]==0 and len(jj)==20 and jj[0]==52: zpar=m.x
 print(h,b,np.array(res).T, 'p',zpar)
coef=np.linalg.lstsq(aff(w),T,rcond=None)[0]; u,sv,v=np.linalg.svd(coef); t=S['eval_times']; print('trajectory coefficients',coef, 'singular vals',sv,'temporal vectors',v[:2]);
for name,f in [('log',lambda p: np.log1p(p[0]*t)),('exp',lambda p:-np.expm1(-p[0]*t)),('sqrt',lambda p:np.sqrt(1+p[0]*t)-1),('rat',lambda p:t/(1+p[0]*t))]:
 for extra in [lambda p:t,lambda p:np.sin(p[1]*t),lambda p:-np.expm1(-p[1]*t),lambda p:np.log1p(p[1]*t),lambda p:np.ones(len(t))]:
  def res(p):
   D=np.stack([f(p),extra(p)]); return (coef-np.linalg.lstsq(D.T,coef.T,rcond=None)[0].T@D).ravel()
  m=least_squares(res,[1.,.22],bounds=(.0001,10));print(name, np.linalg.norm(m.fun),m.x)
print('tail index and w range', np.sort(F@fit(F))[[0,-1]], (a['_readout'](aff(a['_decode'](S['test']['x'],a['_calibration'](S['panel_a'])))@a['_fit'](aff(p['u']),p['v']))@fit(F)).min())
