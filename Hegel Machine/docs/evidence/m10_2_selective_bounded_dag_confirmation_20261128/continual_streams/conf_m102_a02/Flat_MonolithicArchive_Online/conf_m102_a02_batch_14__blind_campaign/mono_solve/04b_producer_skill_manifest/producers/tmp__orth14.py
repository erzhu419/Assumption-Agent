from graph14 import *
from scipy.linalg import orthogonal_procrustes
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.optimize import differential_evolution
B=S['panel_b']; u,v=B['u'],B['v']; tr=S['train']; z=m._inverse(tr['x'],calibrate(S)); Y=np.c_[tr['y'],tr['traj']]; n=52
# Jointly estimate a shared odd profile and an orthogonal mixing matrix.
def get(d):
 dc=d-d.mean(0); vc=v-v.mean(0); R,scale=orthogonal_procrustes(dc,vc); scale/=np.sum(dc**2); return R,scale,v.mean(0)-scale*d.mean(0)@R
R,sc,b=get(u)
for _ in range(25):
 q=((v-b)@R.T).ravel(); a=u.ravel(); order=np.argsort(abs(a)); xx=abs(a)[order]; yy=(q*np.sign(a))[order]; spl=UnivariateSpline(np.r_[0,xx],np.r_[0,yy],s=.00006); d=np.sign(u)*spl(abs(u)); R,sc,b=get(d)
print('orthogonal residual',np.std(sc*d@R+b-v),'range',xx[-1], 'R',R,'bias',b); print('profile',np.round(np.c_[np.arange(0,1.41,.05),spl(np.arange(0,1.41,.05))],6));print('even odd checks',[(t,np.std((v-b)@R.T-t)) for t in [u,d]])
D={'rat':lambda x,p:x/(1+p[0]*x*x),'tanh':lambda x,p:np.tanh(p[0]*x),'atan':lambda x,p:np.arctan(p[0]*x),'sin':lambda x,p:np.sin(p[0]*x),'gauss':lambda x,p:x*np.exp(-p[0]*x*x),'rat3':lambda x,p:x/(1+p[0]*abs(x)**3),'tanh3':lambda x,p:np.tanh(p[0]*x**3),'sin3':lambda x,p:np.sin(p[0]*x**3),'atan3':lambda x,p:np.arctan(p[0]*x**3),'soft':lambda x,p:x/(1+p[0]*abs(x)),'cubic':lambda x,p:x**3,'ratp':lambda x,p:x/(1+p[0]*abs(x)**p[1]),'gaup':lambda x,p:x*np.exp(-p[0]*abs(x)**p[1]),'pow':lambda x,p:np.sign(x)*abs(x)**p[0],'gauss3':lambda x,p:x*np.exp(-p[0]*abs(x)**3)}
q=(v-b)@R.T; models=[]
for names in [(k,) for k in D]+list(itertools.combinations(D,2)):
 def mat(x,p):
  ci=0; out=[x]
  for k in names:
   kk=2 if k in ('ratp','gaup') else 1;out.append(D[k](x,p[ci:ci+kk]));ci+=kk
  return np.column_stack([a.ravel() for a in out])
 def obj(p):
  a=mat(u,p);return np.mean((a@ls(a,q.ravel())-q.ravel())**2)
 bounds=sum(([ (.03,8),(.5,6)] if k in ('ratp','gaup') else [(.03,8)] for k in names),[]); o=differential_evolution(obj,bounds,seed=3,popsize=6,maxiter=45,tol=1e-6); w=ls(mat(u,o.x),q.ravel()); d=(mat(u,o.x)@w).reshape(u.shape); rr,ss,bb=get(d); h=(mat(z,o.x)@w).reshape(z.shape); models.append((np.std(ss*d@rr+bb-v),names,o.x,w,ss*h@rr+bb))
models.sort(key=lambda q:q[0]); print('ORTH MODEL',*[q[:4] for q in models[:20]],sep='\n');np.savez('/tmp/orthops14.npz',z=z,**{f'b{i}':q[-1] for i,q in enumerate(models[:12])},spl=np.sign(z)*spl(abs(z))@R+b)
for idx,(_,names,_,_,h) in enumerate(models[:12]+[(0,('spline',),0,0,np.sign(z)*spl(abs(z))@R+b)]):
 f=feats(np.c_[z,h]); mu=f[:n,1:].mean(0); sig=np.maximum(f[:n,1:].std(0),1e-6); f[:,1:]=(f[:,1:]-mu)/sig; best=[]
 for l in (0,.001,.003,.01,.03,.1):
  w=ls(np.r_[f[:n],np.sqrt(l)*np.eye(f.shape[1])],np.r_[Y[:n],np.zeros((f.shape[1],9))]); er=abs(f@w-Y);best.append((l,er[n:,0].mean(),er[n:,1:].mean()))
 print('downstream',idx,names,np.round(best,5))
