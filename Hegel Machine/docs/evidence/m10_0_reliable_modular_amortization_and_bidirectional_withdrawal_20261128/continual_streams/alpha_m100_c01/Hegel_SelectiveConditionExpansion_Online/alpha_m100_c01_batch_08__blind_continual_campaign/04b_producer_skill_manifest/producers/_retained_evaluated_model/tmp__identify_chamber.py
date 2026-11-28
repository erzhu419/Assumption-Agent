exec(open('/tmp/probe_chamber.py').read().split('for j,xq in enumerate(xqs):')[0])
from scipy.optimize import least_squares
base=np.column_stack((h[:,None]*th[:2],np.zeros(len(h))))@B.T
R=Y-base
for name,x in zip(('joint','train+diag','diag only'),xqs):
 dr=f(x)@th[2:]; H=np.column_stack((np.ones(len(h)),h,dr,z)); slope=np.linalg.lstsq(dr[:,None],R,rcond=None)[0][0]; e=R-dr[:,None]*slope
 print(name,'slope',np.round(slope,5),'rmse by horizon',np.round(np.sqrt(np.mean(e**2,axis=0)),5)); print('nuisance coeff first/mid/last/rate',np.linalg.lstsq(H,R,rcond=None)[0][:,[0,5,11,12]].round(4))
print('times',t)
dr=f(xqs[-1])@th[2:]; s=np.linalg.lstsq(dr[:,None],R,rcond=None)[0][0]; print('intrinsic-mode drift coefs',np.linalg.lstsq(B,s,rcond=None)[0]);
def col(tau):return np.r_[-tau*np.expm1(-t/tau),1.]
models={'linear':(lambda q:np.r_[t,1.],[]),'unit withdrawal':(lambda q:col(1),[]),'one decay':(lambda q:col(np.exp(q[0])),[0.]),'decay and remainder':(lambda q:q[1]*np.r_[t,1.]+(1-q[1])*col(np.exp(q[0])),[0.,.2]),'two decays':(lambda q:q[2]*col(np.exp(q[0]))+(1-q[2])*col(np.exp(q[1])),[-.8,1.2,.3]),'rational log':(lambda q:np.r_[np.log1p(t*np.exp(q[0]))/np.exp(q[0]),1.],[0.]),'rational strain':(lambda q:np.r_[t/(1+t*np.exp(q[0])),1.],[0.])}
for name,(fn,q) in models.items():
 if q:q=least_squares(lambda q:(R-dr[:,None]*fn(q)).ravel(),q,loss='soft_l1',f_scale=.008).x
 E=R-dr[:,None]*fn(q);print(name,q,'slope',np.round(fn(q)[[0,5,11,12]],5),'q50 q90 max curve and rate',np.quantile(abs(E[:,:-1]),[.5,.9,1]),np.quantile(abs(E[:,-1]),[.5,.9,1]))
for l in np.unique(q_lab:=d['chamber_verification_trials']['lab_id']):
 sel=q_lab==l; print('diag ref lab',l,[(np.median(v[sel].mean(0)),np.median(v[sel].std(0))) for v in [d['chamber_verification_trials']['ref_coupons']]],'dr residual rate',np.round((R[:,-1]-dr)[sel],4))
