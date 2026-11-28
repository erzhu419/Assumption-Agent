from wall_probe import *
np.set_printoptions(precision=9,suppress=True)
for d in 'xy':
 k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
 print(d,'h',h,'kappa ends',k[[0,-1]],'vel ends',v[[0,-1]],'first/last face',((k[:-1]+k[1:])/2)[[0,-1]],'wrap mean',np.mean(k[[0,-1]]))
def convert(p,kind):
 p=np.asarray(p).reshape(4,-1)
 if kind=='wrap-only': return np.column_stack([np.ones(4),np.zeros(4),p[:,0]]).ravel()
 if kind=='no-leak': return np.column_stack([p[:,0],np.zeros(4),p[:,1]]).ravel()
 if kind=='full': return p.ravel()
for kind,p0 in [('wrap-only',[1]*4),('no-leak',[1,1]*4),('full',[1,0,1]*4)]:
 for omit in [-1,0,1,2]:
  keep=[i for i in range(3) if i!=omit]
  fit=least_squares(lambda p:1e5*(calc(convert(p,kind))-Y)[:,keep].ravel(),p0,gtol=1e-10)
  p=convert(fit.x,kind); print(kind,omit,'coef',p,'RMS',rms(calc(p)-Y))
  if kind=='no-leak' and omit==-1:
   for d,b in zip('xy',p.reshape(2,2,3)):
    k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
    eff=b[:,[0,2]]*k[[0,-1],None]; print(d,'effective k',eff,'neighbor minus endpoint',eff[:,0]-k[[0,-1]],'wrap over arithmetic mean',eff[:,1]/np.mean(k[[0,-1]]))
