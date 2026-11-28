exec(open('/tmp/analyze_chamber.py').read().split('for s,xx in ')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
y=_response(dg); h=dg['hold_channel_raw'].sum(-1); p=dg['past_channel_raw'].sum(-1); drift=th[2]+xd@th[3:]
A=_design(dg,xd,dt,t,tau); base=A@th
print('baseline diagnostics MAE and p90',np.mean(abs(y-base),axis=0),np.quantile(abs(y-base),.9,axis=0))
def basis(q): return np.column_stack((np.r_[-np.expm1(-t/q[0]),1/q[0]],np.r_[-np.expm1(-t/q[1]),1/q[1]],np.r_[t,1.]))
def fits(q):
 b=basis(q); co=np.linalg.lstsq(b,y.T,rcond=None)[0].T; return co,y-co@b.T
for q in [tau,least_squares(lambda q: fits(q)[1].ravel(),tau,bounds=([.06,.8],[2.,20.]),loss='soft_l1',f_scale=.015).x]:
 co,e=fits(q); print('basis tau',q,'MAE/p90',np.mean(abs(e),axis=0),np.quantile(abs(e),.9,axis=0)); print('h, drift, branch1/branch2/linear excess vs baseline:'); print(np.c_[h,drift,co-np.c_[A[:,0,:2]*th[:2]/(-np.expm1(-t[0]/tau)),drift]])
 dec=np.exp(-dt[:,None]/tau); mem=p@((1-dec)*np.cumprod(np.vstack((dec[1:],np.ones(2)))[::-1],axis=0)[::-1]); cand=np.c_[np.ones(len(h)),h,drift,mem,p,p[:,1:]-p[:,:-1],h-p[:,-1],np.abs(h-p[:,-1])]; print('corr features h,d,mem2,p6,diff5,jump,absjump vs co'); print(np.corrcoef(cand[:,1:].T,co.T)[:-3,-3:]);
