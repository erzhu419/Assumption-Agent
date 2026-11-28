import numpy as np
from itertools import combinations
from scipy.special import erf
q=np.load('/tmp/d11.npz'); b=q['b']; Y=np.c_[q['y'],q['T']]; n=len(Y); norm=np.linalg.norm(b,axis=1); rng=np.random.default_rng(811); ps=[np.arange(n)]+[rng.permutation(n) for _ in range(20)]+[np.argsort(norm)]; rng=np.random.default_rng(31415); extra=[rng.permutation(n) for _ in range(40)]; tail=[np.argsort(s*b[:,k]) for k in range(4) for s in (-1,1)]; split=ps+extra+tail

def gram(fun,k,variant):
 h=fun(b,k); x=np.c_[b,h]; idx=list(combinations(range(8),2)); idx=[(i,j) for i,j in idx if variant=='full' or i%4==j%4 or (i<4<=j) or (variant=='bb' and j<4) or (variant=='hh' and i>=4)]; return np.c_[np.ones(n),x,x*x,np.column_stack([x[:,i]*x[:,j] for i,j in idx])]

def pred(F,ridge):
 K=F@F.T; out=[]
 for p in split:
  i,j=p[:52],p[52:]; basis=np.linalg.svd(Y[i,1:],full_matrices=False)[2][:2]; Z=np.c_[Y[i,0],Y[i,1:]@basis.T]; p=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+ridge*np.eye(len(i)),Z); out.append(np.c_[p[:,0],p[:,1:]@basis]);
 return np.array(out)

def report(name,p):
 E=np.abs(p-np.array([Y[p[52:]] for p in split])); E=np.stack([E[:,:,0].mean(1),E[:,:,1:].mean((1,2)),E[:,:,-1].mean(1)],1); print(name,*[v.round(5) for v in (E[0],E[1:21].mean(0),E[1:21].max(0),E[21],E[22:62].mean(0),E[22:62].max(0),E[62:].mean(0),E[62:].max(0))]);

soft=lambda x,k:x/np.sqrt(1+(k*x)**2); rat=lambda x,k:x/(1+(k*x)**2); tanh=lambda x,k:np.tanh(k*x); atan=lambda x,k:np.arctan(k*x); ef=lambda x,k:erf(k*x)
F0=np.c_[.4*b,np.arcsinh(2*b)]; F0=np.c_[np.ones(n),F0/2,np.column_stack([F0[:,i]*F0[:,j]/(8 if i==j else np.sqrt(32)) for i in range(8) for j in range(i,8)])]; A=pred(F0,2e-7); B=pred(F0,4e-6); A[:,:,1:]=B[:,:,1:]; report('current',A)
P={}; specs=[('soft',soft,1.4,.0003),('rat',rat,1,.003),('tanh',tanh,1.4,.003),('tanh+y',tanh,1.75,.0003),('atan',atan,1.75,.0003),('erf',ef,1.4,.003)]
for name,fun,k,r in specs:
 for v in ('mixed','bb','hh'):
  F=gram(fun,k,v); P[name,v]=pred(F,r); report(name+' '+v,P[name,v]);
for v in ('mixed','bb','hh'):
 M=np.mean([P[name,v] for name,*_ in specs],0); report('ensemble '+v,M);report('ensemble '+v+' 20% old',.8*M+.2*A)
M=np.mean([P[name,'mixed'] for name in ('soft','rat','tanh','erf')],0); M[:,:,0]=(P['tanh+y','mixed'][:,:,0]+P['soft','mixed'][:,:,0]+P['atan','mixed'][:,:,0])/3;report('separate ensemble',M)
for name,fun,k,r in specs[:4]:
 F=gram(fun,k,'mixed'); c=np.linalg.lstsq(F,Y,rcond=1e-9)[0]; print(name,'fit residual',np.mean(abs(F@c-Y),0).round(5),'coef scalar and last',c[:,[0,-1]].round(3).T)
