exec(open('/tmp/probe.py').read().split('for k in')[0])
u=c['u']; print('N',len(x),'split',max(40,len(x)-20),'panel input range',u.min(),u.max())
from scipy.optimize import least_squares
for key in ['v1','v2']:
 v=c[key]
 def proj(A): return A@fit(A,v)-v
 for family in ['sin','tanh','soft','exp','poly3','poly4']:
  def f(q,p):
   if family=='sin': return np.sin(q*p)/p
   if family=='tanh': return np.tanh(q*p)/p
   if family=='soft': return q/np.sqrt(1+(q*p)**2)
   if family=='exp': return np.expm1(q*p)/p
   if family=='poly3': return q+p*q**3
   return q+p*q*q+.15*p*p*q**3
  for aug in [False,True]:
   def feat(q,p):return np.c_[np.ones(len(q)), f(q,p), q*q] if aug else np.c_[np.ones(len(q)), f(q,p)]
   lo=.001 if family in ['sin','tanh','soft'] else -5
   ff=lambda p:proj(feat(u,p)).ravel()
   best=min((least_squares(ff,np.full(4,start),bounds=(lo,6),max_nfev=120) for start in [.05,.5,1.4,2.7]),key=lambda r:r.cost)
   P=feat(u,best.x); H=P@np.linalg.pinv(P); print(key,family,aug,np.round(best.x,3),round(np.std(proj(P)/(1-np.diag(H))[:,None]),6))
