import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/m99_probe.py').read())
from scipy.optimize import least_squares
for key,panel in [('train',tr),('chamber',ch)]:
 y=panel['strain_curves_obs']-panel['strain_0'][:,None]; r=panel['instant_rate_obs']; Y=np.column_stack((y,r));
 def modes(tau):
  F=np.column_stack((-np.expm1(-t[:,None]/tau),t)); F=np.vstack((F,np.r_[1/tau,1])); A=np.linalg.lstsq(F,Y.T,rcond=None)[0].T; return A,Y-A@F.T
 for loss,scale in [('linear',1),('soft_l1',.01),('cauchy',.01)]:
  fit=least_squares(lambda u:modes(np.exp(u))[1].ravel(),np.log(tau),bounds=(np.log(.05),np.log(25)),loss=loss,f_scale=scale); tt=np.sort(np.exp(fit.x)); A,E=modes(tt); print(key,loss,'taus',tt,'closure curve',np.quantile(abs(E[:,:-1]),[.5,.9,.99,1]),'rate',np.quantile(abs(E[:,-1]),[.5,.9,1]));
 np.savez('/tmp/m99_'+key+'_modes.npz',tau=tt,amp=A,resid=E)
print('base intrinsic',tau,theta,'times',t,'dt',dt)
print('calibration residual',np.quantile(abs(raw@b[:-1]+b[-1]-L.sum(1)),[0,.5,.9,1]),'coef',b)
sw=ca['single_channel_sweeps'];
for lab in np.unique(sw['lab_id']):
 use=sw['lab_id']==lab; X=sw['true_stress'][use]; Y=sw['cmd_stress'][use]; print('sweep',lab,np.polynomial.polynomial.polyfit(X,Y,1),np.max(abs(Y-np.polynomial.polynomial.polyval(X,np.polynomial.polynomial.polyfit(X,Y,1)))))
print('archived solver prediction law:')
