exec(open('/tmp/m98_probe.py').read().split("print('baseline chamber'")[0])
np.set_printoptions(precision=5,suppress=True,linewidth=145)
def kern(v): return np.vstack([-np.expm1(-t[:,None]/v),1/v])
K=kern(tau); r=drift@b[2:]; D=Y-(hold[:,None]*b[:2])@K.T
C=D@np.linalg.pinv(K.T)
print('r, measured rate minus reset branches, fitted two branch amplitudes, amp/r:',np.c_[r,D[:,-1],C,C/r[:,None]])
print('intrinsic kernel only',metrics(C@K.T-D), 'SVD',np.linalg.svd(D,compute_uv=False)[:6]); print('end ratios',np.median(D[r>.1]/r[r>.1,None],axis=0))
scale=np.r_[np.repeat(.004,len(t)),.008]; delta=np.r_[t,1.]
def reg(name,B):
 coef=np.linalg.lstsq((B/scale[None,:,None]).reshape(-1,B.shape[-1]),(D/scale).ravel(),rcond=None)[0]; print(name, np.round(coef,5),metrics(B@coef-D))
reg('intrinsic kernels X drift',r[:,None,None]*K[None,:,:]); reg('intrinsic kernels X feature columns',(drift[:,None,:,None]*K[None,:,None,:]).reshape(len(r),len(t)+1,-1))
reg('intrinsic kernels X r and r^2', (np.c_[r,r*r][:,None,:,None]*K[None,:,None,:]).reshape(len(r),len(t)+1,-1))
reg('intrinsic kernels X r and signed square', (np.c_[r,r*abs(r)][:,None,:,None]*K[None,:,None,:]).reshape(len(r),len(t)+1,-1))
for kind in ('exp','log','rational','sqrt'):
 def F(s):
  a=np.exp(s); u=t/a
  vals={'exp':-np.expm1(-u),'log':np.log1p(u),'rational':u/(1+u),'sqrt':2*(np.sqrt(1+u)-1)}[kind]
  return np.r_[a*vals,1.]
 for free in (False,True):
  opt=least_squares(lambda p: ((r[:,None]*F(p[0])*(p[1] if free else 1.)-D)/scale).ravel(),[0.,1.] if free else [0.],bounds=([-5.,-10.],[5.,10.]) if free else ([-5.],[5.]))
  print(kind,free,np.exp(opt.x[0]),opt.x[1:],metrics(r[:,None]*F(opt.x[0])*(opt.x[1] if free else 1.)-D))
print('residual amplitude after r regression, correlations against stress/features:')
err=C-r[:,None]*np.linalg.lstsq(r[:,None],C,rcond=None)[0]; q=np.c_[hold,past,x,r*r,r*abs(r),r**3,hold*r]; print(np.round(np.corrcoef(np.c_[err,q].T)[:2,2:],3))
print('raw calibrated channels hold sample',((ch['hold_channel_raw']-offset)@np.linalg.inv(np.linalg.lstsq(np.c_[np.ones(len(d['calibration_load_cells']['bridge_raw'])),d['calibration_load_cells']['true_channel_loads']],d['calibration_load_cells']['bridge_raw'],rcond=None)[0][1:]))[:4])
