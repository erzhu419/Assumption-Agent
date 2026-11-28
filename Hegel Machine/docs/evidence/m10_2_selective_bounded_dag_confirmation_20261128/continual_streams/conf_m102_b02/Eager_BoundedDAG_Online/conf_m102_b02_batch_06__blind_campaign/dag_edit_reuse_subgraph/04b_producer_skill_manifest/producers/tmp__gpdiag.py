exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
sys.path.insert(0,'/holdout_task');import node_00 as m
z,b,c,p=m.features(s)
a=s['panel_a']; af=m.AF(a['u']); keep=m.cal(s,a['u'])[2]<2; aw=m.fit(af[keep],a['v'][keep]); print('A coef',aw,'noise',np.std(af[keep]@aw-a['v'][keep]),'ranges',[(g.min(0),g.max(0)) for g in (z,b,c,p)], flush=True)
print('inverse residual',np.max(abs(m.AF(z)@aw-np.r_[t['x'],s['test']['x']])),flush=True)
from scipy.spatial.distance import cdist
U=np.linalg.svd(t['traj'],full_matrices=False)[2]; YY=np.c_[T[:,0],t['traj']@U[:2].T]

def gp(f,Y,i,j,kind):
 f=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-9); f=f[:n];Y=(Y-Y[i].mean(0))/Y[i].std(0)
 I=np.eye(len(i)); d=(f[:,None,:]-f[None,:,:])**2; dt=d[j][:,i]; d=d[i][:,i]; aa=f[i]; zz=f[j]; N=aa.shape[1]
 def calc(q,dm,L,R):
  w=np.exp(q[:N]); r2=dm@w; scale=np.exp(q[N]); dot=(L*w)@R.T
  if kind==0: k=scale*np.exp(-.5*r2); dk=-.5*k[:,:,None]*dm*w
  else:
   r=np.sqrt(5*r2); er=np.exp(-r); k=scale*(1+r+r*r/3)*er;dk=-5/6*scale*((1+r)*er)[:,:,None]*dm*w
  poly=np.exp(q[N+1])*(1+dot)**2; lin=np.exp(q[N+2])*np.ones_like(k)
  dk+=2*np.exp(q[N+1])*(1+dot)[:,:,None]*(L[:,None,:]*R[None,:,:])*w
  return k+poly+lin,(dk,k,poly,lin)
 def fun(q):
  k,(dk,k0,p,l)=calc(q,d,aa,aa); noise=np.exp(q[-1]); cf=cho_factor(k+noise*I); Yi=Y[i]; v=cho_solve(cf,Yi); h=Y.shape[1]; g=.5*(h*cho_solve(cf,I)-v@v.T)
  return .5*np.sum(Yi*v)+h*np.log(np.diag(cf[0])).sum(),np.r_[np.einsum('ij,ijk->k',g,dk),np.sum(g*k0),np.sum(g*p),np.sum(g*l),noise*np.trace(g)]
 q=np.r_[np.full(N,-2.3),0.,-2.,0.,-6.]; rr=minimize(fun,q,jac=True,method='L-BFGS-B',bounds=[(-13,4)]*N+[(-12,8),(-15,6),(-12,7),(-14,-1)],options={'maxiter':130}); q=rr.x; k=calc(q,d,aa,aa)[0]; w=cho_solve(cho_factor(k+np.exp(q[-1])*I),Y[i]); pred=calc(q,dt,zz,aa)[0]@w; return pred,np.exp(q[:N]),rr.fun

Fs={'z':z,'zb':np.c_[z,b],'zc':np.c_[z,c],'bc':np.c_[b,c],'zbc':np.c_[z,b,c],'zbcp':np.c_[z,b,c,p],'z0b0c0':np.c_[z,np.sign(z)*np.maximum(abs(z)-.35,0)**1.5,z*z]}
for name,F in Fs.items():
 for kind in [0,1]:
  vals=[]
  for i,j in sl:
   a,w,v=gp(F,YY[:,:1],i,j,kind); o,w2,v2=gp(F,YY[:,1:],i,j,kind); e=a[:,0]*YY[i,0].std()+YY[i,0].mean()-YY[j,0]; E=(o*YY[i,1:].std(0)+YY[i,1:].mean(0))@U[:2]-t['traj'][j]; vals.append([abs(e).mean(),abs(E).mean()]);
   if len(vals)==1: print('ARD',name,kind,w.round(3),w2.round(3),flush=True)
  print('GP',name,kind,np.round(vals,5),flush=True)
