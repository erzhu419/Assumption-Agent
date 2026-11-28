import numpy as np,pickle
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=s['train']['x']; T=s['train']['traj']; y=s['train']['y']; Y=np.c_[y,T]; n=len(X)
A=[]
for u,v in zip(s['panel_a']['u'].T,s['panel_a']['v'].T):
 i,j=np.triu_indices(len(u),1); q=abs(u[i]-u[j])>.08; i,j=i[q],j[q]; slope=np.median((v[i]-v[j])/(u[i]-u[j])); offset=np.median(v-slope*u); keep=abs(v-(offset+slope*u))<.001; A.append(np.linalg.lstsq(np.c_[np.ones(sum(keep)),u[keep]],v[keep],rcond=None)[0])
A=np.array(A); Z=(X-A[:,0])/A[:,1]; p=s['panel_b']; B=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; H=(Z-B[0])@np.linalg.inv(B[1:]); G=Z@B[1:]+B[0];
R=np.random.default_rng(827); splits=[(np.arange(52),np.arange(52,72))]+[(r[:52],r[52:]) for r in [R.permutation(n) for _ in range(12)]]+[(r[:52],r[52:]) for r in [np.argsort(X[:,i]*sign) for i in range(4) for sign in [-1,1]]];
def score(P):
 e=abs(P-np.array([Y[j] for i,j in splits])); z=np.c_[e[:,:,0].mean(1),e[:,:,1:].mean((1,2)),e[:,:,1:].mean(1).max(1)]; return np.vstack((z[0],z[1:13].mean(0),z[1:13].max(0),z[13:].mean(0),z[13:].max(0)))
def bases(z):
 g=z@z.T; q=(z[:,None]-z[None,:])**2; d=q.sum(2);
 return {'p2':g**2,'p3':g**3,'p4':g**4,'p5':g**5,'rb1':np.exp(-d/2),'rb2':np.exp(-d/8),'rb3':np.exp(-d/32),'mat':(1+np.sqrt(5*d)+5*d/3)*np.exp(-np.sqrt(5*d)),'add':np.exp(-q/2).sum(2),'sqadd':(np.exp(-q/2).sum(2))**2,'diag3':z**3@(z**3).T,'diag4':z**4@(z**4).T,'sqrt':(np.sqrt(1+.924*z*z)-1)@(np.sqrt(1+.924*z*z)-1).T}
basesall={k:bases(z) for k,z in [('x',X),('a',Z),('b',G),('bi',H)]}
base=2+4*(X@X.T); out=[]
for v,D in basesall.items():
 for keys in [('p2','p3'),('p2','p3','p4','p5'),('p2','rb1'),('p2','rb2'),('p2','rb3'),('p2','mat'),('p2','add','sqadd','rb2'),('p2','p3','diag3','diag4','sqrt','rb2')]:
  mats=np.array([D[key] for key in keys]+[np.eye(n)]); P=[]; scales=[]
  means=np.sqrt(np.mean(mats**2,axis=(1,2))); means[-1]=1; mats/=means[:,None,None];
  for i,j in splits:
   U,S,V=np.linalg.svd(T[i],full_matrices=False); reduced=np.c_[y[i],U[:,:2]*S[:2]]; co=[]
   C=mats[:,i][:,:,i]; Bij=base[i][:,i]; Cj=mats[:,j][:,:,i]; Bji=base[j][:,i];
   for col in range(3):
    t=reduced[:,col];
    def obj(th):
     w=np.exp(th); K=Bij+np.einsum('i,ijk->jk',w,C); L=cho_factor(K); a=cho_solve(L,t); inv=cho_solve(L,np.eye(len(i))); der=(inv-a[:,None]*a[None,:]); return .5*t@a+np.log(np.diag(L[0])).sum(),.5*np.einsum('jk,ijk,i->i',der,C,w)
    opt=minimize(obj,[-3]*(len(keys))+[-9],method='L-BFGS-B',jac=True,bounds=[(-17,3)]*len(keys)+[(-13,-3)],options={'maxiter':80}); w=np.exp(opt.x); K=Bij+np.einsum('i,ijk->jk',w,C); Kj=Bji+np.einsum('i,ijk->jk',w,Cj); co.append(Kj@cho_solve(cho_factor(K),t));
   P.append(np.c_[co[0],np.column_stack(co[1:])@V[:2]]);
  P=np.array(P); out.append((v,keys,score(P),P));
def ranking(z): return (z*np.array([.3,1,.6,.9,.35])[:,None]).mean()
for entry in sorted(out,key=lambda h:ranking(h[2]))[:16]:print(entry[0],entry[1],np.round(entry[2],4).tolist())
for v in ['x','a','b','bi','all']:
 entries=[p for name,k,e,p in out if (name==v or v=='all') and k[1] in ('p3','rb1','rb2','rb3','mat')];print('ensemble',v,np.round(score(np.mean(entries,axis=0)),4).tolist())
