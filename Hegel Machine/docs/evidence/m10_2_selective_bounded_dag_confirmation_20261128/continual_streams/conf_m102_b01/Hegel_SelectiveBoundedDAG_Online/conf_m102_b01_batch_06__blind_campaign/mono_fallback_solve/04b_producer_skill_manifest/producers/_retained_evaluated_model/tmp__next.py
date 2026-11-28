from graph import *
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True,linewidth=135)
a=s['panel_c']; u=a['u']; print('C coefficients',ls(poly(u,2),a['v1']));print('C errs',np.std(poly(u,2)@ls(poly(u,2),a['v1'])-a['v1'],0));
b=s['panel_b']; u=b['u']; v=b['v']; tests=[('cubic rational',lambda u,p:u+p[0]*u**3/(1+abs(p[1])*u*u),[-2,3]),('linear rational',lambda u,p:u+p[0]*u/(1+abs(p[1])*u*u),[-1.1,2]),('tanh',lambda u,p:u+p[0]*np.tanh(p[1]*u),[-.6,2]),('atan',lambda u,p:u+p[0]*np.arctan(p[1]*u),[-.6,2]),('erf',lambda u,p:u+p[0]*__import__('scipy').special.erf(p[1]*u),[-.6,2]),('gauss',lambda u,p:u+p[0]*u*np.exp(-abs(p[1])*u*u),[-1.1,2]),('cub+quint',lambda u,p:u+p[0]*u**3+p[1]*u**5,[2,-1]),('tanh+cub',lambda u,p:u+p[0]*np.tanh(p[1]*u)+p[2]*u**3,[-.6,2,0]),('sin+tanh',lambda u,p:u+p[0]*np.sin(p[1]*u)+p[2]*np.tanh(p[3]*u),[-.4,2.9,.1,2.])];
for name,f,p in tests:
 def r(p):
  F=aff(f(u,p));return (F@ls(F,v)-v).ravel()
 o=least_squares(r,p,max_nfev=350);print(name,o.x,np.std(o.fun))
tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; n=52; z,A,B,C,D=ops(s,tr['x']);b=B(z);e=C(b);d=D(z); f=D(b); P=np.c_[e,d];
for name,F in [('Cmodels',np.c_[aff(e),poly(d,2)[:,5:],d]),('epsD',aff(P)),('quad',poly(P,2)),('combined',np.c_[poly(P,2),b,z])]:
 Q=ls(F[:n],Y[:n]); print(name,'mae',np.mean(abs((F@Q-Y)[:n]),0),np.mean(abs((F@Q-Y)[n:]),0)); order=np.argsort(-np.linalg.norm(Q*F[:n].std(0)[:,None],axis=1)); names=['']*len(Q) if name!='quad' else [str(i) for k in range(3) for i in itertools.combinations_with_replacement(range(8),k)]; print([(i,names[i],np.round(Q[i,[0,1,4,8]],3).tolist()) for i in order[:16]])
T=Y[:,1:]; t=s['eval_times']; U,S,V=np.linalg.svd(T,full_matrices=False); print('SVD',S,V[:3]);
for name,F in [('exp',lambda a:1-np.exp(-a*t)),('log',lambda a:np.log1p(a*t)),('sat',lambda a:t/(1+a*t)),('pow',lambda a:t**a)]:
 def loss(a):
  f=F(a);return np.linalg.norm(T-np.outer(T@f,f)/(f@f))
 o=minimize_scalar(loss,bounds=(.01,6),method='bounded'); print('time',name,o.x,o.fun,F(o.x)/F(o.x)[-1])
