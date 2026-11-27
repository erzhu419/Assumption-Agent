import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/m99_probe.py').read())
q=np.load('/tmp/m99_chamber_modes.npz'); tt,A=q['tau'],q['amp']; p,h,D=cc; X=D[:,1:]; labs=ch['lab_id']; Y=np.column_stack((ch['strain_curves_obs']-ch['strain_0'][:,None],ch['instant_rate_obs']))
def z(u, pp=p, dd=dt):
 a=np.exp(-dd[:,None]/np.atleast_1d(u)); return pp@((1-a)*np.cumprod(np.vstack((np.ones(a.shape[1]),a[:0:-1])),axis=0)[::-1])
print('effective taus',tt,'amp ranges',np.quantile(A,[0,.5,1],axis=0),'amp/hold ranges',np.quantile(A[:,:2]/h[:,None],[0,.5,1],axis=0))
base=np.column_stack((D,h,p)); names=['one','x0','x1','x2','x3','hold']+['p%d'%j for j in range(6)]
def regress(M):
 B=np.linalg.lstsq(M,A,rcond=None)[0]; E=M@B-A
 cv=np.empty_like(A)
 for lab in np.unique(labs):
  trn=labs!=lab; cv[~trn]=M[~trn]@np.linalg.lstsq(M[trn],A[trn],rcond=None)[0]-A[~trn]
 return np.round(B,4),np.round(np.sqrt(np.mean(E**2,axis=0)),4),np.round(np.max(abs(cv),axis=0),4)
print('linear full',names,regress(base));
for label,M in [('hold',h[:,None]),('affine hold',base[:,[0,5]]),('hold+past',base[:,[0,5,6,7,8,9,10,11]]),('feature+hold',base[:,:6]),('states intrinsic',np.column_stack((D,h,z(tau)))),('states effective',np.column_stack((D,h,z(tt)))),('states reverse',np.column_stack((D,h,z(tau,p[:,::-1],dt[::-1])))),('lags and pairwise',np.column_stack((base[:,[0,5,6,7,8,9,10,11]],(p[:,1:]-p[:,:-1])**2))),('stress and pair products',np.column_stack((base[:,[0,5,6,7,8,9,10,11]],h[:,None]*p)))]:
 B,err,cv=regress(M); print(label,'rms',err,'worst LOLO',cv,'coef',B.tolist() if M.shape[1]<=8 else '')
print('amps and intrinsic states');print(np.round(np.column_stack((h,z(tau),z(tt),A)),4))
for name,W in [('train',tr),('test',te)]:
 print(name,'aux mean/std',np.mean(W['aux_chamber_telemetry'],axis=0),np.std(W['aux_chamber_telemetry'],axis=0))
