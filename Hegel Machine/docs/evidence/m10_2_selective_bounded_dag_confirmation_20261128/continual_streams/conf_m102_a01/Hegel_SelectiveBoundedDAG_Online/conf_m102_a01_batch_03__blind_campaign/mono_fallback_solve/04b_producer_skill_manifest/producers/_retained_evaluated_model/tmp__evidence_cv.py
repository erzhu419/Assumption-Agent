from explore import *
import glob, importlib.util
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize, nnls
from sklearn.model_selection import KFold
spec=importlib.util.spec_from_file_location('prior',glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
D={}; W={'x':x,'z':z,'h':h,'Bz':z@B[1:]+B[0],'Bx':x@B[1:]+B[0],'ix':(x-B[0])@np.linalg.inv(B[1:])}
for name,g in W.items():
 p=poly(g,6); D[name]=[p[:,l:r] for l,r in zip([5,15,35,70,126],[15,35,70,126,210])]
 D[name+'t']=[poly(np.tanh(.7*g),2)[:,1:],poly(np.tanh(1.25*g),2)[:,1:]]
 D[name+'r']=[poly(g/(1+.22*g*g),2)[:,1:]]

def evidence(i,j,Y,groups,default=-6,reg=.1):
 base=10+10*z@z.T; ii=np.ix_(i,i); ji=np.ix_(j,i); I=np.eye(len(i)); G=[]
 for f in groups:
  f=f/np.maximum(np.sqrt(np.mean(f[i]**2,axis=0)),1e-8); G.append(f@f.T/f.shape[1]);
 A=np.array([g[ii] for g in G]+[I]); C=np.array([g[ji] for g in G]); Y=Y[i]; mu=np.array([default]*len(G)+[-8]);
 def loss(p):
  w=np.exp(p); c=cho_factor(base[ii]+np.einsum('i,ijk->jk',w,A)); a=cho_solve(c,Y); Z=cho_solve(c,I)-np.outer(a,a); return .5*Y@a+np.log(c[0].diagonal()).sum()+reg*np.sum((p-mu)**2),.5*w*np.einsum('ijk,jk->i',A,Z)+2*reg*(p-mu)
 b=minimize(loss,mu,jac=True,bounds=[(-17,2)]*len(G)+[(-13,-4)],method='L-BFGS-B',options={'maxiter':100}).x; w=np.exp(b); return (base[ji]+np.einsum('i,ijk->jk',w[:-1],C))@cho_solve(cho_factor(base[ii]+np.einsum('i,ijk->jk',w,A)),Y),w

def pred(i,j):
 ss={**s,'train':{k:s['train'][k][i] for k in ['x','y','traj']},'test':{'x':x[j]}}; p,k,v=prior._components(ss); out={'prior':.65*p[:,0]+.35*k[:,0],'p':p[:,0],'k':k[:,0]}
 for name in W:
  out['ep'+name],w=evidence(i,j,y,D[name]);
  for deg in [3,4,5]:
   for lam in [.4,2,8]:
    fs=[aff(z)]+D[name][:deg-1]; f=np.concatenate([f/np.sqrt(np.mean(f[i]**2,axis=0))/np.sqrt(l) for f,l in zip(fs,[1e-7,.035,lam*.2,lam,lam*6,lam*20])],axis=1); out[name+str(deg)+'_'+str(lam)]=f[j]@f[i].T@np.linalg.solve(np.eye(len(i))+f[i]@f[i].T,y[i])
 for name,groups in [('multi',[f for k in W for f in D[k][:3]]),('fwd',[f for k in ['z','Bx','Bz'] for f in D[k][:4]]),('mixed',[f for k in ['z','Bx','h'] for f in D[k][:4]]),('nonlinear',[f for k in W for f in D[k][:3]+D[k+'t']+D[k+'r']])]:
  out[name],w=evidence(i,j,y,groups)
  if len(i)==52 and min(j)==52: print('weights',name,np.round(w,4))
 return out
P=[]; inds=[]
for seed in [19,23,29]:
 for i,j in KFold(5,shuffle=True,random_state=seed).split(np.arange(52)):
  P.append(pred(i,j)); inds.extend(j)
gate=pred(np.arange(52),np.arange(52,72)); names=list(gate); A=np.column_stack([np.concatenate([p[k] for p in P]) for k in names]); Y=y[inds]; C=np.column_stack([gate[k] for k in names]); E=np.mean(abs(A-Y[:,None]),0); J=np.mean(abs(C-y[52:,None]),0); idx=np.argsort(E)
for ix in [idx[:15],np.argsort(J)[:12],np.array([names.index(k) for k in ['prior','p','k','multi','mixed','nonlinear']])]: print([(names[j],round(E[j],5),round(J[j],5)) for j in ix]);
for ids in [idx[:4],idx[:10],idx[:20],np.arange(len(names))]:
 for reg in [.0,.05,.2,1.]:
  w=nnls(np.r_[A[:,ids],np.ones((1,len(ids)))*10,np.eye(len(ids))*reg],np.r_[Y,10,np.zeros(len(ids))])[0]; w/=w.sum();print('mix',len(ids),reg,np.mean(abs(A[:,ids]@w-Y)),np.mean(abs(C[:,ids]@w-y[52:])),[(names[k],round(t,4)) for k,t in zip(ids,w) if t>.01]);
np.savez('/tmp/predictions_cv.npz',A=A,C=C,Y=Y,names=names,inds=inds)
