import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from pathprobe import *
from gp import gp
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
_,_,vt=np.linalg.svd(t[:52],full_matrices=False); T=t@vt[:2].T
print('metadata, corr, fit',s['eval_times'],s['telemetry_vec'],np.corrcoef(np.c_[y,T].T),np.linalg.lstsq(np.c_[np.ones(52),T[:52]],y[:52],rcond=None)[0]); print('panel A diag res',np.sort(np.linalg.norm(s['panel_a']['v']-s['panel_a']['u']*a-b,axis=1)))
for kind in (0,1,4):
 for k in ('z','Bz','Iz'):
  es=[]; q=D[k];
  for i,j in splits:
   h,p,l=gp(q[i],q[j],y[i],kind); es.append(np.mean(abs(h[:,0]-y[j])))
  print('GP',kind,k,np.round(es,5))

def model(q,i,j,mode,path):
 coords=[q, D[path]]; dist=lambda u:cdist(u,u,'sqeuclidean'); f1=coords[0]; f2=coords[1]; E=dist(f1); E2=dist(f2); ii,jj=np.triu_indices(4); K0=1+f1@f1.T; P=[np.ones_like(E),f1@f1.T,(f1[:,ii]*f1[:,jj])@(f1[:,ii]*f1[:,jj]).T]; P.extend(np.exp(-E/h) for h in (1,3,8,18,38)); P.extend(np.exp(-E2/h) for h in (.5,1.5,4,10,25)); P.append(f2@f2.T);
 if mode>=2:
  for key in ('z','Bz','Iz'):
   v=D[key];
   for k in (.4,1,2):
    f=np.tanh(k*v); f2=v*v/(1+k*v*v); P.extend([f@f.T,f2@f2.T,(f@f.T+1)**2])
 P=np.array(P); H=P[:,i[:,None],i]; Hq=P[:,j[:,None],i]; Y=y[i]; V=t[i]@np.linalg.svd(t[i],full_matrices=False)[2][:2].T; E0=np.eye(len(i)); A=np.c_[np.ones(len(i)),V] if mode%2 else np.ones((len(i),1));
 def fun(log,ret=False):
  w=np.exp(log); k=np.einsum('k,kij->ij',w[:-1],H)+w[-1]*E0; fac=cho_factor(k); av=cho_solve(fac,A); b=np.linalg.solve(A.T@av,A.T@cho_solve(fac,Y)); a=cho_solve(fac,Y-A@b)
  if ret:return a,b,w
  C=cho_solve(fac,E0)-np.outer(a,a); return .5*(Y-A@b)@a+np.log(np.diag(fac[0])).sum(), .5*w*np.r_[np.einsum('ij,kij->k',C,H),np.trace(C)]
 h=minimize(fun,np.r_[np.full(len(P),-4.),-8.],jac=True,method='L-BFGS-B',bounds=[(-15,8)]*len(P)+[(-13,-4)],options={'maxiter':150}).x; alpha,b,w=fun(h,True); vtnew=np.linalg.svd(t[i],full_matrices=False)[2][:2]; tq,_,_=gp(q[i],q[j],t[i]@vtnew.T,0); pred=np.einsum('k,kij->ij',w[:-1],Hq)@alpha+b[0]+(tq@b[1:] if mode%2 else 0); return pred,b,w
for m in range(4):
 for key in ('z','0Bz','B0z','0z','1Bz','1z','1Iz'):
  es=[]
  for i,j in splits:
   h,be,w=model(D['z'],i,j,m,key); es.append(np.mean(abs(h-y[j])));
  print('LMC',m,key,np.round(es,5),np.round(be,3),flush=True)
