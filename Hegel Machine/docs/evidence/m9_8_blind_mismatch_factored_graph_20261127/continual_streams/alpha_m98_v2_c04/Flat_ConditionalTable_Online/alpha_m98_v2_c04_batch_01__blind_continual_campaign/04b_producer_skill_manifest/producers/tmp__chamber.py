exec(open('/tmp/baseline.py').read().split('for name,b,xx')[0])
from scipy.optimize import least_squares
N,K=yd.shape
sig=.006
# Profile coupon amplitudes and drift while finding shared chamber time constants.
def basis(q):
 u=np.exp(q); return np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.]))
y=np.c_[yd,rd]; q0=np.log(tau)
def decomp(q):
 A=basis(q); c=np.linalg.lstsq(A,y.T,rcond=None)[0].T; return c,c@A.T-y
r=least_squares(lambda q:decomp(q)[1].ravel(),q0,bounds=(np.log([.06,1.]),np.log([1.,50.])))
print('free chamber taus',np.exp(r.x),'bench',tau,'rmse free and fixed',*[np.sqrt(np.mean(decomp(q)[1]**2,axis=0)) for q in (r.x,q0)])
np.set_printoptions(precision=5,suppress=True,linewidth=160)
for q in (q0,r.x):
 c,e=decomp(q); u=np.exp(q); print('chamber coefficients',c); print('time residuals',e.mean(0),e.max(0),e.min(0)); print('largest residual indexes',np.argwhere(abs(e)>.035).tolist());
 z=lambda v:pd@(-np.expm1(-dt[:,None]/v)*np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0][:,None]/v)); dd=th[2]+xd@th[3:]; za=z(tau); zb=z(u); D=np.c_[hd,pd,za,zb,xd,dd]; names=['hold',*[f'p{i}' for i in range(len(dt))],'za0','za1','zb0','zb1',*[f'x{i}' for i in range(4)],'bench_drift']; tgt=np.c_[c[:,:2]/th[:2],c[:,2],c[:,2]-dd]; print('corr',list(zip(names,np.corrcoef(D.T,tgt.T)[:-4,-4:].round(3))));
 for j in range(2):
  for tag,zj in [('bench',za[:,j]),('chamber',zb[:,j])]:
   A=np.c_[np.ones(N),hd,zj]; b=np.linalg.lstsq(A,tgt[:,j],rcond=None)[0]; print('gap fit',j,tag,b,np.sqrt(np.mean((A@b-tgt[:,j])**2)))
 for tag,A in [('x',np.c_[np.ones(N),xd]),('x+load',np.c_[np.ones(N),xd,hd,za]),('drift',np.c_[np.ones(N),dd]),('drift+load',np.c_[np.ones(N),dd,hd,za])]:
  b=np.linalg.lstsq(A,c[:,2],rcond=None)[0]; print('drift fit',tag,b,np.sqrt(np.mean((A@b-c[:,2])**2)))
print('sweeps',d['calibration_load_cells']['single_channel_sweeps'])
