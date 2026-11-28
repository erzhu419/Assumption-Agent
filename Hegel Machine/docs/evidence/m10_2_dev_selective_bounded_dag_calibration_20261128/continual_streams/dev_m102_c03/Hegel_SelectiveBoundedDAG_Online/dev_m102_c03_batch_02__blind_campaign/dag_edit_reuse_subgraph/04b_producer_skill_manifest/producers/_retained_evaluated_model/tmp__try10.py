import os,runpy,pickle,numpy as np
from scipy.optimize import least_squares,minimize
from itertools import combinations,combinations_with_replacement
from scipy.linalg import cho_factor,cho_solve
b=runpy.run_path('/tmp/base9.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; z=b['_calibrate'](s,a['x']); y=a['y']; n=len(y); ii=np.arange(n); splits=[(ii<52,ii>=52),(ii>=20,ii<20)]+[(ii%4!=j,ii%4==j) for j in range(4)]; p=s['panel_c']; u,v=p['u'],p['v1']; h=b['_linear']
forms=[('r2',lambda x,p:x/(1+p[0]*abs(x)+p[1]*x*x),[.9,.2]),('r3',lambda x,p:x/(1+p[0]*abs(x)+p[1]*x*x+p[2]*abs(x)**3),[.9,.2,.01]),('mixsin',lambda x,p:x/(1+p[0]*abs(x))+p[1]*np.sin(p[2]*x),[1,-.1,1.2]),('pow',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.45,1,2])];
for nm,f,g in forms:
 def err(q):
  F=h(f(u,q));return (F@np.linalg.lstsq(F,v,rcond=None)[0]-v).ravel()
 res=least_squares(err,g,max_nfev=300); print('v1',nm,res.x,np.std(res.fun))
print('n',n)

def families(z):
 s=np.sin(1.2*z); r=z/(1+.9*abs(z)+.2*z*z); t=np.tanh(1.2*z); out=[]
 for x in z.T:
  S=np.sin(1.2*x);T=np.tanh(1.2*x);R=x/(1+.9*abs(x)+.2*x*x)
  out+=[np.c_[x,S,T,R,np.arctan(1.8*x)],np.c_[x*x,S*S,T*T,R*R,np.exp(-x*x),x**4]]
 for i,j in combinations(range(4),2):
  x,w=z[:,i],z[:,j];si,sj=s[:,i],s[:,j];ri,rj=r[:,i],r[:,j];ti,tj=t[:,i],t[:,j]
  out += [np.c_[x*w,si*sj,ri*rj,ti*tj,x*w/(1+.5*(x*x+w*w)),np.cos(1.2*(x+w)),np.cos(1.2*(x-w))],np.c_[si**2*sj,ri**2*rj,ti**2*tj,x*x*w],np.c_[si*sj**2,ri*rj**2,ti*tj**2,x*w*w],np.c_[si**2*sj**2,ri**2*rj**2,ti**2*tj**2,x*x*w*w],np.c_[np.sin(1.2*(x+w)),np.sin(1.2*(x-w)),np.tanh(x+w),np.tanh(x-w)]]
 for inds in combinations(range(4),3):out+=[np.c_[np.prod(s[:,inds],1),np.prod(r[:,inds],1),np.prod(t[:,inds],1),np.prod(z[:,inds],1)]]
 return out

def kernels(z,mode):
 base=b['_dictionary'](z,s); g=families(z)
 if mode=='family':return g
 if mode=='union':return g+base
 if mode=='single':return [f[:,i:i+1] for f in g for i in range(f.shape[1])]
 if mode=='hybrid':return g+[f[:,i:i+1] for f in g[:8] for i in range(f.shape[1])]+base[:6]

def bayes(F,i,j,mode,noise):
 Y=y[i]; mu=Y.mean(); sd=Y.std(); Y=(Y-mu)/sd; N=len(Y); I=np.eye(N); K=[];L=[]
 for f in F:
  f=f-f[i].mean(0); f=f/np.maximum(np.sqrt(np.mean(f[i]**2,axis=0)),1e-8)/np.sqrt(f.shape[1]); K.append(f[i]@f[i].T);L.append(f[j]@f[i].T)
 K=np.array(K).reshape(len(K),-1);L=np.asarray(L); M=len(K)
 def ev(p,pred=False):
  w=np.exp(p); c=cho_factor((w[:-1]@K).reshape(N,N)+w[-1]*I,check_finite=False); A=cho_solve(c,I,check_finite=False); al=A@Y
  if pred:return mu+sd*(w[:-1]@L.reshape(M,-1)).reshape(-1,N)@al
  if mode=='ml':
   D=A-np.outer(al,al); val=.5*Y@al+np.log(np.diag(c[0])).sum()
  else:
   d=A.diagonal(); e=al/d; val=np.mean(e*e); G=-2/N*((e/d)[:,None]*(A*al[None,:])- (e*e/d)[:,None]*A*A); D=G+G.T
   return val,.5*w*np.r_[K@D.ravel(),np.trace(D)]
  return val,.5*w*np.r_[K@D.ravel(),np.trace(D)]
 p=minimize(ev,np.r_[np.full(M,-4.),-6.],jac=True,method='L-BFGS-B',bounds=[(-16,6)]*M+[(noise,-1)],options={'maxiter':140,'ftol':1e-8}).x
 return ev(p,True)
for f in ['family','hybrid','union','single']:
 F=kernels(z,f)
 for criterion in ['ml','loo']:
  for noise in [-11,-7,-5.5]:
   pred=[bayes(F,i,j,criterion,noise) for i,j in splits]; print(f,criterion,noise,np.round([np.abs(P-y[j]).mean() for P,(_,j) in zip(pred,splits)],5),flush=True)
