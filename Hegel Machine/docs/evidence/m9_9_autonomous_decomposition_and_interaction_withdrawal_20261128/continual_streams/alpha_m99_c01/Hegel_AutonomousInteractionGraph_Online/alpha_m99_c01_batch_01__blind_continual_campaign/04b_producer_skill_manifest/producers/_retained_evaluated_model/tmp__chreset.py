exec(open('/tmp/chbasis.py').read().split('C,E=profile(tau)')[0])
_,xs=m._features(di,di)
r=beta[2]+xs@beta[3:]; P=basis(tau); C,E=profile(tau)
print('RESET residual amplitudes vs hold-only, rows r,dA0,dA1,D')
print(np.round(np.c_[r,C[:,:2]-hd[:,None]*beta[:2],C[:,2]],5))
for name,q in [('r',r),('rplus',np.maximum(r,0)),('softplus',np.logaddexp(0,r)),('tanh',np.tanh(r)),('exp',np.exp(r)),('x',xs)]:
 F=np.column_stack((np.ones(len(r)),q)); base=(hd[:,None]*beta[:2])@P[:,:2].T; D=Y-base; B=np.linalg.lstsq(F,D,rcond=None)[0]; h=(np.linalg.qr(F)[0]**2).sum(1); e=(D-F@B)/(1-h[:,None]); print(name,'LOO RMS/max',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.max(abs(e[:,:-1])),np.max(abs(e[:,-1])),'mode coef',np.linalg.lstsq(P,B.T,rcond=None)[0].T.round(5))
print('test telemetry summary',np.round(np.c_[np.mean(te['aux_chamber_telemetry'],axis=0),np.std(te['aux_chamber_telemetry'],axis=0)],3))
