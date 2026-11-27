exec(open('/tmp/probe_chamber.py').read().split('for tag,')[0])
from itertools import combinations_with_replacement
q=np.r_[t,1.]; Y=np.c_[yc,rc]; n=len(Y); age=np.cumsum(dt[::-1])[::-1]-dt
z=pc@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); r=beta[2]+xc@beta[3:]; u=np.array([tau[0],tau[1]*1.055]); T=np.c_[np.r_[-np.expm1(-t[:,None]/u),[1/u]],q]; amp=np.linalg.lstsq(T,Y.T,rcond=None)[0].T
P=np.c_[hc,z,hc[:,None]-z,pc,r,xc,d['coupon_features_raw']]; names='h z0 z1 g0 g1 p0 p1 p2 p3 p4 p5 r x0 x1 x2 x3 raw0 raw1 raw2 raw3'.split(); P=np.where(np.isfinite(P),P,0.)
for j,core in enumerate((np.c_[np.ones(n),hc,r],np.c_[np.ones(n),hc,r],np.c_[np.ones(n),r])):
 M=np.eye(n)-core@np.linalg.pinv(core); e=M@amp[:,j]; scored=[]
 def check(name,v):
  mv=M@v; den=np.linalg.norm(mv)*np.linalg.norm(e)
  if den>1e-9: scored.append((abs(mv@e/den),name))
 for i,name in enumerate(names):
  check(name,P[:,i]); check('tanh('+name+')',np.tanh(P[:,i])); check('abs('+name+')',abs(P[:,i]));
 for i,k in combinations_with_replacement(range(len(names)),2):
  check(names[i]+'*'+names[k],P[:,i]*P[:,k]); check(names[i]+'/(1+'+names[k]+'^2)',P[:,i]/(1+P[:,k]**2));
 print('branch',j,'residual by lab',[(int(l),np.round(e[d['lab_id']==l],4).tolist()) for l in np.unique(d['lab_id'])]); print('best correlations',sorted(scored,reverse=True)[:15]);
print('lab, imputed, hold, states, features, drift, branch coeff'); print(np.round(np.c_[d['lab_id'],np.isnan(d['coupon_features_raw']).any(1),hc,z,xc,r,amp],4))
