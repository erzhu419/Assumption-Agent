from rheo_probe import *
from scipy.optimize import least_squares
D=th[2]+xg@th[3:]; L=g['lab_id']; Y=yc-base; R=yr-rbase; E=-np.expm1(-t[:,None]/tau)
M=pg@(-np.expm1(-dt[:,None]/tau)*np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0][:,None]/tau)); G=(hg[:,None]-M)*th[:2]
print('D, late/D, onset residual, gaps'); print(np.round(np.c_[D,Y[:,-1]/D,R,G],4))
raw=c['bridge_raw']; z=np.c_[np.ones(len(raw)),raw]@b-c['true_channel_loads']; print('bridge RMSE',np.sqrt(np.mean(z**2,0)))
print('time-by-time residual slope/intercept',np.round(np.linalg.lstsq(np.c_[np.ones(len(D)),D],Y,rcond=None)[0],5))
print('rate residual correlations',[(k,np.round(np.corrcoef(v,R,rowvar=False)[-1,:-1],3)) for k,v in [('D X',np.c_[D,xg]),('gaps hold lab',np.c_[G,hg,L]),('past',pg)]])
for label,f in [('rat2',lambda k:t/(1+k*t*t)),('atan',lambda k:np.arctan(k*t)/k),('erf',lambda k:__import__('scipy').special.erf(k*t)*np.sqrt(np.pi)/(2*k)),('stretch',lambda k:t*np.exp(-k*t)),('quad',lambda k:t-k*t*t),('cubic',lambda k:t-k*t**3),('sat',lambda k:-np.expm1(-k*t)/k)]:
 def predict(q):return base+D[:,None]*(f(np.exp(q[0]))-t)
 scores=[]
 for l in [None,*np.unique(L)]:
  m=np.ones(len(L),bool) if l is None else L!=l
  q=least_squares(lambda q:(predict(q)[m]-yc[m]).ravel(),[-1],loss='soft_l1',f_scale=.008).x
  err=predict(q)-yc
  if l is None: full=[float(np.exp(q[0])),float(np.sqrt(np.mean(err**2)))]; print(label,full,'median late bias',round(np.median(err[:,-1]),5))
  else:scores.append(np.sqrt(np.mean(err[~m]**2)))
 print(' worst lab',max(scores))
