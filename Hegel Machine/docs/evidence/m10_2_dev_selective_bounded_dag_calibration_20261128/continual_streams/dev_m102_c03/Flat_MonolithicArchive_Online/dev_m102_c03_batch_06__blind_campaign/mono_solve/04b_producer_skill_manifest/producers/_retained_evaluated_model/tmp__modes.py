import numpy as np,itertools as it,pickle
from scipy.optimize import least_squares
D=dict(np.load('/tmp/expanded.npz')); z=D['z']; Y=D['Y']; Q=D['Q']; n=52
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']['traj']; t=s['eval_times']; aff=lambda x:np.c_[np.ones(len(x)),x]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]; cy=lambda x:x*np.roll(x,-1,axis=1)
c=aff(cy(z))@D['C']; d=aff(np.cos(2.4*z))@D['D']; p=cy(z); h=np.cos(2.4*z); zz=np.stack([z[:,i]*z[:,j] for i,j in it.combinations_with_replacement(range(4),2)],1); z3=np.stack([np.prod(z[:,i],1) for i in it.combinations_with_replacement(range(4),3)],1)
for name,X in {'cd':np.c_[c,d],'cdot':np.c_[c,d,(c*d).sum(1)],'cdiag':np.c_[c,d,c*d],'coff':np.c_[c,d,c[:,0]*d[:,1]],'mixed':np.c_[c,d,c*d,c[:,0]*d[:,1]],'poly3':np.c_[z,zz,z3],'mixed3':np.c_[z,c,d,c*d,c[:,0]*d[:,1],z3]}.items():
 X=aff(X); b=fit(X[:n],Y[:n]); e=abs(X@b-Y); print(name,e[:n].mean(0).round(3),e[n:].mean(0).round(3),b.round(3).tolist() if name=='mixed' else '')
# Screen coherent low-dimensional nonlinear nodes by improvement conditional on the archived readout.
base=np.c_[c,d,c*d,c[:,0]*d[:,1]]; pool={}
for name,x in dict(z=z,c=c,d=d,p=p,h=h,cd=c*d,cs=c.sum(1,keepdims=True),ds=d.sum(1,keepdims=True),dot=(c*d).sum(1,keepdims=True),zsq=z*z).items():
 for j,w in enumerate(x.T):pool[name+str(j)]=w
for a,b in [('c','d'),('p','h'),('z','c'),('z','d'),('c','c'),('d','d')]:
 A=np.column_stack([pool[a+str(j)] for j in range(4)]); B=np.column_stack([pool[b+str(j)] for j in range(4)])
 for i,j in it.product(range(4),repeat=2):
  for sym,w in [('+',A[:,i]+B[:,j]),('-',A[:,i]-B[:,j]),('*',A[:,i]*B[:,j])]:pool[f'{a}{i}{sym}{b}{j}']=w
names=[]; F=[]
for k,x in pool.items():
 for nm,g in [('id',x),('sq',x*x),('cube',x**3),('rat',x/(1+.22*x*x)),('evenrat',1/(1+.22*x*x)),('tanh',np.tanh(x)),('softplus',np.logaddexp(0,x)),('abs',abs(x))]+[(f'{a}{w}',(np.sin if a=='sin' else np.cos)(w*x)) for a in ('sin','cos') for w in (.5,1,1.25,1.8,2.4,3.0,4.8)]:names.append(k+'_'+nm);F.append(g)
F=np.array(F).T
for bname,B in [('main',base),('extended',np.c_[base,z,zz]),('bare',np.empty((len(z),0)))]:
 X=aff(B); V=np.linalg.qr(X[:n])[0]; E=Y[:n]-V@(V.T@Y[:n]); G=F[:n]-V@(V.T@F[:n]); gn=np.linalg.norm(G,axis=0); S=abs(E.T@G)/np.where(gn>1e-6,gn,np.inf)
 for j in [0,1,2,3]:
  ind=S[j].argsort()[-9:][::-1]; print(bname,j,[(names[i],round(S[j,i]/np.linalg.norm(E[:,j]),3)) for i in ind]);
  if bname!='bare':
   for k in ind[:1]:
    M=np.c_[X,F[:,k]]; e=abs(M@fit(M[:n],Y[:n,j])-Y[:,j]); print('mae',e[:n].mean(),e[n:].mean())
