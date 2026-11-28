from check_batch import m,s,tr,idx
import numpy as np
from scipy.optimize import least_squares
fit=m._fit
z=m._calibrate(s['panel_a'],tr['x']); c,w=m._operators(s['panel_c'],z)
B=m._aff(np.c_[m._quadratic(z),c*w]); t=s['eval_times']; u=s['panel_b']['u']; b=m._aff(z)@fit(m._aff(u),s['panel_b']['v'])
radius=np.linalg.norm(z,axis=1); rs=np.argsort(radius); rng=np.random.default_rng(771)
splits=[idx,rs]+[rng.permutation(idx) for _ in range(8)]
raw={'z':z,'b':b,'z+w':z+w,'z-w':z-w,'z+c':z+c,'z-c':z-c,'edge':m._edges(z),'w':w,'c':c}
raw['pair']=np.column_stack([z[:,i]+sg*z[:,j] for i in range(4) for j in range(i+1,4) for sg in (-1,1)])

def variations(i,j,ab):
 yield 'baseline', B[j]@fit(B[i],ab[i])
 for name in ('z','b','z+w','z-w','z+c','edge','w','c','pair'):
  q=raw[name]; sd=q[i].std(0); x=(q-q[i].mean(0))/sd
  for kind,q in [('cubic',x**3),('tanh',np.tanh(x)),('rational',x/(1+x*x)),('cos',np.cos(x)),('smoothabs',np.sqrt(.25+x*x)),('splines',np.column_stack([np.maximum(0,x-k)**2 for k in (-.75,0,.75)]))]:
   q=q-B@fit(B[i],q[i]); q=q/np.maximum(q[i].std(0),1e-8)
   for lam in (1,8,32):
    R=ab[i,0]-B[i]@fit(B[i],ab[i,0]); cc=np.linalg.solve(q[i].T@q[i]+lam*np.eye(q.shape[1]),q[i].T@R)
    pred=B[j]@fit(B[i],ab[i]); pred[:,0]+=q[j]@cc
    yield f'{name}:{kind}:{lam}',pred
 # Fit one nonlinear projection after removing all linear design terms.
 U=np.linalg.qr(B[i],mode='reduced')[0]; y=ab[i,0]; y=y-U@(U.T@y); scale=z[i].std(0); v=z/scale
 for name,fun in [('tanh',np.tanh),('rational',lambda x:x/(1+x*x)),('cos',np.cos),('smoothabs',lambda x:np.sqrt(1+x*x)),('softplus',lambda x:np.logaddexp(0,x))]:
  def obj(p):
   f=fun(v[i]@p[1:5]+p[5]); r=f-U@(U.T@f)
   return np.r_[p[0]*r-y,.0003*p]
  seeds=[np.r_[.3,k,.15] for k in np.vstack((np.eye(4),-np.eye(4),[.8,.6,0,0],[.8,-.6,0,0],[.2,.2,.7,.7]))]
  cand=min((least_squares(obj,p,bounds=(-6,6),max_nfev=120) for p in seeds),key=lambda q:q.cost); p=cand.x
  f=p[0]*fun(v@p[1:5]+p[5]); pred=B[j]@fit(B[i],ab[i]); pred[:,0]+=f[j]-B[j]@fit(B[i],f[i]); yield 'index:'+name,pred

out={}
for k,ids in enumerate(splits):
 i,j=ids[:52],ids[52:]; D=m._basis(t,m._rates(t,tr['traj'][i])); ab=fit(D,tr['traj'].T).T
 for name,pred in variations(i,j,ab):
  e=abs(pred@D.T-tr['traj'][j]);out.setdefault(name,[]).append([e.mean(),e.mean(0).max()])
for name,rows in sorted(out.items(),key=lambda p:np.asarray(p[1])[:,0].max())[:18]+[(q,out[q]) for q in ('baseline','index:tanh','index:rational','index:cos','index:smoothabs','index:softplus')]:
 a=np.array(rows);print(name,'ordered',a[0].round(5),'radial',a[1].round(5),'random max',a[2:].max(0).round(5),'random mean',a[2:].mean(0).round(5))
print('train radius',np.quantile(radius,[0,.5,.72,.9,1]),'test radius',np.quantile(np.linalg.norm(m._calibrate(s['panel_a'],s['test']['x']),axis=1),[0,.5,.72,.9,1]))
