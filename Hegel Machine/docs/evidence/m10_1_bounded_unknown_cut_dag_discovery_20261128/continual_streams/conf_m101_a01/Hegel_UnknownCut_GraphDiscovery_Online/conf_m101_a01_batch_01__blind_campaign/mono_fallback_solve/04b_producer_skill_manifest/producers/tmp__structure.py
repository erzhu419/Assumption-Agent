from graph_probe import *
from scipy.optimize import least_squares
np.set_printoptions(precision=3,suppress=True)
q=frames(x)['a-']; n=len(q); L=poly(q,1); P=poly(q,2); T=poly(q,3); rng=np.random.default_rng(7)
for k in ('v1','v2'):
 p=s['panel_c']; u=p['u']; v0=p[k]; J=np.c_[u, u*u, np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)]), np.sin(u),np.tanh(u),u**3]; print(k)
 for j in range(4):
  base=np.c_[np.ones(len(u)),u]; r=v0[:,j]-base@np.linalg.lstsq(base,v0[:,j],rcond=None)[0]; selected=[]
  for _ in range(4):
   rr=[]
   for h in range(4,J.shape[1]):
    d=np.c_[base,J[:,selected+[h]]]; rr.append((np.std(v0[:,j]-d@np.linalg.lstsq(d,v0[:,j],rcond=None)[0]),h))
   score,h=min(rr);selected.append(h);print(j, h,round(score,5),end='; ')
  print()
for j,g in enumerate(Y.T):
 print('target',j,'polynomial coefficients',np.linalg.lstsq(T,g,rcond=None)[0]);
 for f in ('tanh','sin','rat','soft','exp','tanh2','trig'):
  def basis(t,inds):
   Z=L[inds]@t.reshape(-1,5).T
   if f.startswith('tanh'): h=np.tanh(Z)
   elif f=='sin': h=np.sin(Z)
   elif f=='trig': h=np.concatenate([np.sin(Z),np.cos(Z)],axis=1)
   elif f=='rat': h=1/(1+Z*Z)
   elif f=='soft': h=np.logaddexp(Z,-Z)
   else: h=np.exp(np.clip(Z,-9,9))
   return np.c_[P[inds],h]
  results=[]
  for a,b in splits[:4]:
   def resid(t,ret=False):
    d=basis(t,a); c=np.linalg.lstsq(d,g[a],rcond=1e-8)[0]; return (c,basis(t,b)@c) if ret else d@c-g[a]
   winner=None
   for i in range(12):
    ini=rng.normal(0,.8,10 if f=='tanh2' else 5); opt=least_squares(resid,ini,bounds=(-5,5),max_nfev=140); err=np.mean(abs(resid(opt.x))); c,p=resid(opt.x,True)
    if winner is None or err<winner[0]:winner=(err,np.mean(abs(p-g[b])),opt.x,c)
   results.append(winner)
  print(f,[(round(r[0],4),round(r[1],4)) for r in results], 'plane and weights',results[0][2], results[0][3])
