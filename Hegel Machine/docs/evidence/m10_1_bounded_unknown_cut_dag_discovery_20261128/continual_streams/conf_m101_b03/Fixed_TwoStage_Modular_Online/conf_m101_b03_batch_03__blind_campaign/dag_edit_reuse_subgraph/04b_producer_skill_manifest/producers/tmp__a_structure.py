import itertools as it,pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a']; u=p['u']; v=p['v']; A=np.c_[np.ones(len(u)),u]; w=np.ones(len(u))
for _ in range(25):
 B=np.linalg.lstsq(A*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(A@B-v,axis=1); w=1/np.hypot(1,r/np.median(r))
keep=np.argsort(r)[:-3]; u,v,A=u[keep],v[keep],A[keep]; pairs=list(it.combinations(range(4),2)); W=np.stack([u[:,i]*u[:,j] for i,j in pairs],1); triples=np.stack([np.prod(u[:,q],1) for q in it.combinations(range(4),3)],1)
def fit(g,y=v):
 D=np.c_[A,g]; B=np.linalg.lstsq(D,y,rcond=None)[0]; e=D@B-y; return np.mean(e**2),B,e/(1-np.diag(D@np.linalg.pinv(D))).reshape((-1,)+(1,)*(e.ndim-1))
fs={'sin':lambda x,k:np.sin(k*x)/k,'tanh':lambda x,k:np.tanh(k*x)/k,'rat':lambda x,k:x/(1+k*x*x),'asinh':lambda x,k:np.arcsinh(k*x)/k,'atan':lambda x,k:np.arctan(k*x)/k}
for name,f in fs.items():
 def fun(k):return fit(f(W,k))[0]
 k=minimize_scalar(fun,bounds=(.001,3),method='bounded').x; a,B,e=fit(f(W,k)); print(name,k,np.sqrt(a),np.sqrt(np.mean(e**2,0)),np.round(B[5:],3).T)
G={'product':W,'pair sq':W**2,'pair cube':W**3,'triple':triples,'sq':u*u,'cube':u**3,'sin u':np.sin(2*u),'cos u':np.cos(2*u)}
for tag,f in fs.items():
 for k in (.3,.6,.9,1.2,1.5,2,2.5,3):G[f'{tag}:{k}']=f(W,k)
for i,j in pairs:
 for sign in (-1,1):
  z=u[:,i]+sign*u[:,j]; G[f'sum {i} {j} {sign}']=np.c_[np.sin(z),np.sin(2*z),np.cos(z),np.cos(2*z),np.tanh(z)]
D=np.column_stack(list(G.values())); names=[(k,j) for k,g in G.items() for j in range(g.shape[1])]
for j,y in enumerate(v.T):
 selected=[]
 for _ in range(6):
  O=np.c_[A,D[:,selected]]; Q=np.linalg.qr(O,mode='reduced')[0]; E=D-Q@(Q.T@D); score=np.abs(E.T@y)/np.maximum(1e-10,np.linalg.norm(E,axis=0)); score[selected]=-1; m=score.argmax(); selected.append(m); err,b,loo=fit(D[:,selected],y); print(j,names[m],round(np.sqrt(err),6),round(np.sqrt(np.mean(loo**2)),6),b[5:].round(3))
for name,g in [('prod+triple',np.c_[W,triples]),('prod+cube',np.c_[W,W**3]),('prod+sq',np.c_[W,W**2]),('prod+u nonlinear',np.c_[W,u*u,u**3])]:
 a,b,e=fit(g); print(name,np.sqrt(a),np.sqrt(np.mean(e**2,0)),b[5:].round(3).T)
