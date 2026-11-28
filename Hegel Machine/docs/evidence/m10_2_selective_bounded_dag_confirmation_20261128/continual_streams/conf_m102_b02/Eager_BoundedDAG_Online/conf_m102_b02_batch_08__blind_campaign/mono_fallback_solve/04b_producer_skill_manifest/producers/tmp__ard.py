from features import *
from itertools import combinations,combinations_with_replacement
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
P=lambda v:v*np.roll(v,-1,axis=1)
pr=s['panel_c']; u=pr['u']; WC=fit(D(P(u)),pr['v1']); E=lambda x:D(np.sqrt(1+.924227*x*x)+.831839*x); WE=fit(E(u),pr['v2']);
p=P(b); cb=D(p)@WC; eb=E(b)@WE

def build(z,b,c,e,cb,eb):
 p=P(b); src=dict(z=z,b=b,c=c,e=e,p=p,cb=cb,eb=eb); fs={}
 for name,v in list(src.items()):
  for fn,f in [('id',lambda x:x),('square',np.square),('cube',lambda x:x**3),('abs',abs),('sin',np.sin),('tanh',np.tanh),('asinh',np.arcsinh),('rat',lambda x:x/(1+x*x))]:
   for k in ([1.] if fn in ['id','square','cube','abs'] else [1.25,2.5,5.]):
    fs[f'{name}:{fn}:{k}']=f(k*v)
 for k in [.176,.22,.275,.5,1.25]:
  for name,v in [('z',z),('e',e),('eb',eb),('cb',cb)]:
   for sg in [1,-1]:
    w=b+sg*k*v; key=f'b{sg:+d}{k}{name}'; fs[key+'p']=P(w)
    for n,f in [('sin',np.sin),('tanh',np.tanh),('asinh',np.arcsinh),('rat',lambda x:x/(1+x*x))]:fs[key+n]=f(1.25*w)
 for a,h in combinations_with_replacement(['z','b','p'],2):
  f=src[a];v=src[h];fs[a+'*'+h]=np.einsum('ni,nj->nij',f,v).reshape(len(z),-1)
 return fs

def solve_cov(K,y0,pen=.03,maxiter=100):
 n=len(y0);I=np.eye(n);m=len(K);KF=K.reshape(m,-1); norm=max(np.std(y0),.01);y0=y0/norm
 def loss(s):
  w=np.exp(s);A=(w@KF).reshape(n,n);A.flat[::n+1]+=1e-8
  f=cho_factor(A,check_finite=False);u=cho_solve(f,y0,check_finite=False); inv=cho_solve(f,I,check_finite=False);Q=inv-np.outer(u,u);return .5*(y0@u+2*np.log(f[0].diagonal()).sum())+pen*np.sum(w),w*(.5*(KF@Q.ravel())+pen)
 st=np.full(m,-np.log(m));st[0]=0;st[-1]=-4
 r=minimize(loss,st,jac=True,method='L-BFGS-B',bounds=[(-16,5)]*m,options={'maxiter':maxiter,'ftol':1e-8});return np.exp(r.x)

def kernels(z,b,c,e,cb,eb,i,wide=True):
 fs=build(z,b,c,e,cb,eb); n=len(z); Ks=[np.ones((n,n))]; names=['const']
 for key,v in fs.items():
  w=v-v[i].mean(0); w=w/np.maximum(.005,w[i].std(0));
  if wide:
   for j in range(w.shape[1]):Ks.append(np.outer(w[:,j],w[:,j]));names.append((key,j))
  else:Ks.append(w@w.T/w.shape[1]);names.append(key)
 for key,v in dict(z=z,b=b,p=P(b),cb=cb,eb=eb,e=e,zb=np.c_[.4*z,b],zbp=np.c_[.25*z,b,1.5*cb],q=np.c_[.22*z,b,1.2*D(P(b+.45*e))@WC]).items():
  w=v-v[i].mean(0); w=w/max(.025,np.sqrt(np.mean(w[i]**2))); L=w@w.T/w.shape[1]; d=np.maximum(L.diagonal()[:,None]+L.diagonal()-2*L,0)
  for k in [.125,.5,1.5,5.]:Ks.append(np.exp(-k*d));names.append((key,'rbf',k))
  for k in [2,3]:Ks.append((1+L)**k);names.append((key,'poly',k))
 Ks.append(np.eye(n));names.append('noise');return np.array(Ks),names

def pred(K,i,j,Y,w):
 A=np.einsum('ijk,i->jk',K,w);return A[np.ix_(j,i)]@np.linalg.solve(A[np.ix_(i,i)]+1e-8*np.eye(len(i)),Y[i])
if __name__=='__main__':
 for wide in [False,True]:
  for i,j in splits:
   K,nm=kernels(z,b,c,e,cb,eb,i,wide); w=solve_cov(K[:,i][:,:,i],y[i],maxiter=230); yh=pred(K,i,j,y,w); print('ARD',wide,round(abs(yh-y[j]).mean(),6),'noise',round(w[-1],6),[(nm[k],round(w[k],4)) for k in np.argsort(w)[-12:][::-1]],flush=True)
 print('WB',fit(D(np.sign(s['panel_b']['u'])*np.maximum(abs(s['panel_b']['u'])-.35,0)**1.5),s['panel_b']['v']).round(4),'WC',WC.round(4),'WE',WE.round(4));print('svd',np.linalg.svd(g-g.mean(0),compute_uv=False));print('linear trajectory fit residuals')
 for v in (z,b,np.c_[z,p],np.c_[z,b,p],np.c_[z,p,z*z,p*p],np.c_[z,p,eb]):
  A=D(v);print([np.abs(A[j]@fit(A[i],g[i])-g[j]).mean().round(5) for i,j in splits]);
