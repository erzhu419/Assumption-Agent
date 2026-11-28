exec(open('/tmp/ablate.py').read().split('res=[]')[0]); from scipy.optimize import least_squares
E=[]
for cn,cc in [('raw',C),('tanh',np.tanh(1.25*C)),('atan',np.arctan(1.25*C))]:
 for dn,dd in [('raw',D),('tanh',np.tanh(1.25*D)),('atan',np.arctan(1.25*D))]:
  for bname,b in [('none',z[:,:0]),('z',z),('B',B)]:
   f=np.c_[cc,.2*dd,F(cc,dd),.25*b]
   for r in (.003,.03,.15):
    for typ in ('direct','sinh','cubic','atan','asinh'):
     for gain in ([0.] if typ=='direct' else [.25,.5,1.]):
      ee=[]
      for tr,te in inds:
       mu=f[tr].mean(0); x=m._aff((f-mu)/np.maximum(f[tr].std(0),.0001)); W=x[tr]; G=np.linalg.solve(W.T@W+np.diag(np.r_[0.,np.full(x.shape[1]-1,r)]),W.T); y=Y[tr];
       if typ=='sinh': t=np.sinh(gain*y)/gain; back=lambda p:np.arcsinh(gain*p)/gain
       elif typ=='asinh':t=np.arcsinh(gain*y)/gain;back=lambda p:np.sinh(gain*p)/gain
       elif typ=='cubic':t=y+gain*y**3; back=lambda p:2/np.sqrt(3*gain)*np.sinh(np.arcsinh(1.5*np.sqrt(3*gain)*p)/3)
       elif typ=='atan':t=np.arctan(gain*y)/gain;back=lambda p:np.tan(np.clip(gain*p,-1.4,1.4))/gain
       else:t=y;back=lambda p:p
       pred=back(x[te]@G@t); e=abs(pred-Y[te]);ee.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
      e=np.array(ee);E.append(((cn,dn,bname,r,typ,gain),e))
for j in range(3):
 print('links',j)
 for n,e in sorted(E,key=lambda v:(v[1].mean(0)+.5*v[1].max(0))[j])[:8]: print(n,e[0].round(4),e.mean(0).round(4),e.max(0).round(4))
