import numpy as np,pickle
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=np.load('/tmp/probe.npz')['u']; p=np.load('/tmp/struct6.npz'); n=len(z); t=s['train']['traj']; Y=np.column_stack((s['train']['y'],t@np.linalg.svd(t,full_matrices=False)[2][:2].T));
def kernels(z,p):
 D={}; grams=[]; names=[]
 def add(k,h):
  h=np.asarray(h).reshape(len(z),-1); h=h-h.mean(0); sc=np.sqrt(np.mean(h*h,axis=0)); h=h/np.maximum(sc,1e-8); D[k]=h
 def push(k,g): names.append(k); grams.append(g/np.mean(np.diag(g)))
 for name,x in [('u',z),*p.items()]:
  add(name,x)
  for j,col in enumerate(x.T):
   for tag,f in [('sin',lambda q:np.sin(2.7*q)),('tanh',lambda q:np.tanh(2*q)),('even',lambda q:q*q),('cube',lambda q:q**3)]: add(name+str(j)+tag,f(col))
   for l in (.25,.55,1.):
    w=(col[:,None]-col)/max(np.std(col),.01); push(name+str(j)+'rbf'+str(l),np.exp(-.5*(w/l)**2))
  w=D[name]@D[name].T/x.shape[1]; push(name+'poly2',w**2); push(name+'poly3',w**3)
  for l in (.7,1.4,2.8):
   dd=np.diag(w); push(name+'rbf'+str(l),np.exp(-.5*np.maximum(dd[:,None]+dd-2*w,0)/(l*l)))
 for a,b in combinations(['u',*p],2):
  x=D[a]; v=D[b]; push(a+'*'+b,(x@x.T)*(v@v.T)); add(a+'diag'+b,x*v)
 for k,h in D.items():push(k,h@h.T)
 return np.asarray(grams),np.asarray(names)
K,names=kernels(z,{k:p[k] for k in p.files});
def fit(K,Y,tr,te,prior=.035):
 U=Y[tr]; mean=U.mean(0); scale=U.std(0); U=(U-mean)/scale
 B=K[:,tr][:,:,tr].copy(); shape=B.shape; B=B.reshape(len(B),-1).T.copy(); d=len(B.T); I=np.eye(len(tr)); one=np.ones_like(I); nt=U.shape[1]
 def obj(v):
  w=np.exp(v); A=(B@w[:-1]).reshape(shape[1:])+w[-1]*I+one*3; C=cho_factor(A); a=cho_solve(C,U); ai=cho_solve(C,I); J=nt*ai-a@a.T; val=.5*np.sum(a*U)+nt*np.log(np.diag(C[0])).sum(); g=np.r_[B.T@J.ravel(),np.trace(J)]*.5*w; return val+prior*np.sum(w[:-1]),g+np.r_[prior*w[:-1],0.]
 v=minimize(obj,np.r_[np.full(d,-5.),-5.],jac=True,method='L-BFGS-B',bounds=[(-17,5)]*d+[(-12,0)],options={'maxiter':240,'ftol':3e-7}).x; w=np.exp(v); g=np.einsum('i,ijk->jk',w[:-1],K)+3; a=cho_solve(cho_factor(g[np.ix_(tr,tr)]+I*w[-1]),U); return (g[np.ix_(te,tr)]@a)*scale+mean,w
if __name__=='__main__':
 for tr,te in [(np.arange(52),np.arange(52,72)),(np.r_[0:18,36:72],np.arange(18,36)),(np.arange(18,72),np.arange(18))]:
  for split in (False,True):
   pp=[]
   for ids in ([slice(None)] if not split else [slice(j,j+1) for j in range(3)]):
    pred,w=fit(K,Y[:,ids],tr,te); pp.append(pred); sel=np.argsort(w[:-1])[-8:][::-1]; print('weights',round(w[-1],5),list(zip(names[sel],w[sel].round(3))))
   pred=np.column_stack(pp); state=pred[:,1:]@np.linalg.svd(t,full_matrices=False)[2][:2]; print('mae',tr[0],te[0],split,np.mean(abs(pred[:,0]-Y[te,0])),np.mean(abs(state-t[te])),np.mean(abs(pred-Y[te]),0))
