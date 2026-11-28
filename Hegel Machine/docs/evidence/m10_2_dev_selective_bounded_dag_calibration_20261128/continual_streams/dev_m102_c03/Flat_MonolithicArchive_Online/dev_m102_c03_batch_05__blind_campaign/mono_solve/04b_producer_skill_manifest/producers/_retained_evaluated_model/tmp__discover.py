import numpy as np,pickle,sys
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_m102_c03_batch_03_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; t=s['eval_times']; traj=tr['traj']; n=52
fit=lambda d,y:np.linalg.lstsq(d,y,rcond=1e-9)[0]
aff=lambda u:np.c_[np.ones(len(u)),u]
poly=lambda u:np.column_stack([aff(u)]+[np.prod(u[:,j],axis=1) for j in cwr(range(4),2)])
a=s['panel_a']; u,v=a['u'],a['v']; z=old._calibrate(a,x); print('old inverse sanity',np.quantile(np.linalg.norm(old._calibrate(a,v)-u,axis=1),[0,.25,.5,.75,.9,.95,1]));
for sign in [1,-1]:
 for k in ['p','neighborcube','nbrprod','quadratic','cubic','sin','tanh']:
  def F(u,j):
   w,p=u[:,j],u[:,(j+sign)%4]; terms={'p':[w*p,w**3],'neighborcube':[w*p,p**3],'nbrprod':[w*p,w**3,p**3],'quadratic':[w*p,w*w,p*p],'cubic':[w*p,w*w,p*p,w**3,p**3,w*w*p,w*p*p],'sin':[w*p,np.sin(1.2*w),np.sin(1.2*p)],'tanh':[w*p,np.tanh(w),np.tanh(p)]}[k]; return np.c_[np.ones(len(u)),w,p,*terms]
  mask=np.ones(len(u),bool)
  for _ in range(12):
   co=[fit(F(u,j)[mask],v[mask,j]) for j in range(4)]; err=np.linalg.norm(np.column_stack([F(u,j)@co[j] for j in range(4)])-v,axis=1); mask=err<=max(.001,3*np.median(err))
  print('cal',sign,k,np.quantile(err,[.5,.8,.9]),np.round(np.array(co).T,4) if np.median(err)<.002 else '')
c=s['panel_c']; u=c['u']; B=fit(aff(s['panel_b']['u']),s['panel_b']['v']); C=fit(poly(u),c['v1']); print('C1 coefficients',np.round(C,4))
fun={'cos':lambda q:np.cos(q),'sin':lambda q:np.sin(q),'gauss':lambda q:np.exp(-q*q),'tanh':np.tanh,'soft':lambda q:q/(1+abs(q)),'lorentz':lambda q:1/(1+q*q),'abs':lambda q:abs(q),'exp':lambda q:np.exp(q),'rationalodd':lambda q:q/(1+q*q),'sinc':lambda q:np.sinc(q/np.pi)}
models=[]
for ki,f in fun.items():
 for baseline in [0,1]:
  def D(u,p):return np.c_[aff(u) if baseline else np.ones(len(u)),f(u*p)]
  def residual(p):
   A=D(u,p);return (A@fit(A,c['v2'])-c['v2']).ravel()
  best=min([least_squares(residual,np.full(4,p),bounds=(.04,6),max_nfev=80) for p in [.4,1.2,2.4,3.8]],key=lambda sol:sol.cost); models.append((best.cost,ki,baseline,best.x));
print('C2 top',sorted(models,key=lambda a:a[0])[:10]);
q=lambda p:-np.expm1(-t[:,None]*p); Q=q(least_squares(lambda p:(q(p)@fit(q(p),traj[:n].T)-traj[:n].T).ravel(),[.42,1.83]).x); yy=np.c_[tr['y'],fit(Q,traj.T).T]
for coord,z0 in [('oldcal',z),('raw',x)]:
 b=aff(z0)@B
 for base in [z0,b]:
  c1=poly(base)@C
  for cost,ki,bl,p in sorted(models,key=lambda a:a[0])[:4]:
   D=lambda u:np.c_[aff(u) if bl else np.ones(len(u)),fun[ki](u*p)]; c2=D(base)@fit(D(u),c['v2']); cross=(c1[:,:,None]*c2[:,None,:]).reshape(len(x),16);
   for label,F in [('basic',np.c_[aff(z0),c1,c2]),('diag',np.c_[aff(z0),c1,c2,cross[:,::5].sum(1)]),('full',np.c_[aff(z0),c1,c2,cross])]:
    P=F[n:]@fit(F[:n],yy[:n]); print(coord,'B' if base is b else 'I',ki,label, np.round([np.mean(abs(P[:,0]-yy[n:,0])),np.mean(abs(P[:,1:]@Q.T-traj[n:]))],4))
