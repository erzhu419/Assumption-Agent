from dag_probe import *
from scipy.optimize import least_squares
from itertools import combinations
Q=basis(u,2); n=52; c=C['u']; V=C['v1']
A=basis(c,1); L=np.linalg.lstsq(A,V,rcond=None)[0]; U,S,Vh=np.linalg.svd(L[1:]); print('panel linear',L.round(3))
for name,f,k0 in [('tanh',lambda x,k: np.tanh(k[0]*x),[2.3]),('rational',lambda x,k:x/(1+k[0]*x*x),[2]),('atan',lambda x,k:np.arctan(k[0]*x),[2]),('softsign',lambda x,k:x/np.sqrt(1+k[0]*x*x),[2]),('double',lambda x,k:np.tanh(k[0]*x)+k[1]*np.sin(k[2]*x),[2.3,.2,3])]:
 def fitc(k,i):
  F=np.c_[A,f(c,k)]; return F@np.linalg.lstsq(F[i],V[i],rcond=None)[0]
 k=least_squares(lambda k:(fitc(k,slice(19))[:19]-V[:19]).ravel(),k0,bounds=(.0001,20)).x; print(name,k,np.mean(abs(fitc(k,slice(19))-V),axis=0).round(5),np.mean(abs(fitc(k,slice(19))[19:]-V[19:])))
 for lab,X in [('u',u),('z',z)]:
  H=f(X,k)
  for label,F in [('direct',H),('zdiag',z*H),('udiag',u*H),('zall',(z[:,:,None]*H[:,None,:]).reshape(len(u),-1)),('uall',(u[:,:,None]*H[:,None,:]).reshape(len(u),-1)),('both',np.c_[H,z*H,u*H]),('all',np.c_[H,(u[:,:,None]*H[:,None,:]).reshape(len(u),-1)])]:
   print(name,lab,label,fit(np.c_[Q,F],Y,n))
D={}
for lab,X in [('u',u),('z',z)]:
 for nm,H in [('tanh',np.tanh(2.35*X)),('rat',X*X/(1+.1665*X*X)),('cos',np.cos(X)),('sin',np.sin(X)),('cub',X**3)]: D[lab+nm]=H
for alpha in [1e-5,.0003,.003,.03,.3,3]:
 def sc(F):
  E=F-Q@np.linalg.lstsq(Q[:n],F[:n],rcond=None)[0]; sigma=np.maximum(.01,np.sqrt(np.mean(E[:n]**2,axis=0))); E=E/sigma; r=Y-Q@np.linalg.lstsq(Q[:n],Y[:n],rcond=None)[0]; pred=E@np.linalg.solve(E[:n].T@E[:n]+alpha*np.eye(E.shape[1]),E[:n].T@r[:n]);return round(np.mean(abs((pred-r)[n:])),5)
 for nm,F in [('nonlin',np.column_stack(list(D.values()))),('poly',basis(u,4)[:,15:]),('tensor',np.column_stack([np.c_[H,(u[:,:,None]*H[:,None,:]).reshape(len(u),-1)] for H in D.values()])),('sat tensor',np.column_stack([np.c_[H,(u[:,:,None]*H[:,None,:]).reshape(len(u),-1)] for j,H in D.items() if 'tanh' in j or 'rat' in j]))]:print('ridge',alpha,nm,sc(F))
