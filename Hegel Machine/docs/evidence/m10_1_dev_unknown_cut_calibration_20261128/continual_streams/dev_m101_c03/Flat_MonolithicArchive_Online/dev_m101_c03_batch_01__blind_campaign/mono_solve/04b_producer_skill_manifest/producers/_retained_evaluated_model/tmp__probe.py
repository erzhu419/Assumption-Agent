import numpy as np,pickle,sys
from scipy.optimize import least_squares
from scipy.linalg import svdvals
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(u):return np.c_[np.ones(len(u)),u]
ijs=[(i,j) for i in range(4) for j in range(i,4)]
def quad(u):return np.column_stack([aff(u),*[u[:,i]*u[:,j] for i,j in ijs]])
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a']; b=s['panel_b']; c=s['panel_c']; m=ls(aff(a['u']),a['v']); keep=np.linalg.norm(aff(a['u'])@m-a['v'],axis=1)<2; m=ls(aff(a['u'][keep]),a['v'][keep]); print('A',m)
B=ls(aff(b['u']),b['v']); C=ls(quad(c['u']),c['v1']); print('B',B,'\nC1 coeff',C,'\npairs',ijs,'\nC2 coeff',ls(quad(c['u']),c['v2']))
u,v=c['u'],c['v2']
for label,fun in [('tanh',lambda p:np.tanh(p*u)),('sin',lambda p:np.sin(p*u)),('rational',lambda p:u/(1+p*u*u)),('tanh+lin',lambda p:np.tanh(p[0]*u)+p[1]*u),('sin+lin',lambda p:np.sin(p[0]*u)+p[1]*u),('tanh+quad',lambda p:np.tanh(p[0]*u)+p[1]*u*u),('sin+quad',lambda p:np.sin(p[0]*u)+p[1]*u*u),('rational+quad',lambda p:u/(1+p[0]*u*u)+p[1]*u*u)]:
 def err(p):
  f=aff(fun(p)); return (f@ls(f,v)-v).ravel()
 p=least_squares(err,[1,.2] if '+' in label else [1],max_nfev=120).x; print(label,p,np.std(err(p)))
T=s['train']['traj']; t=s['eval_times']; Y=s['train']['y'];
def F(p):return -np.expm1(-t[:,None]*p)
def errt(p):return (F(p)@ls(F(p),T.T)-T.T).ravel()
for p in [[.3,2],[.6,1.4],[1,3]]:
 p=least_squares(errt,p,bounds=([.01,.01],[12,12])).x; print('rates',p,'noise',np.std(errt(p)))
Z=ls(F(p),T.T).T; W=np.c_[Z,Y]; print('amp sv',svdvals(W),'range',W.min(0),W.max(0)); x=s['train']['x']; z=(x-m[0])/np.diag(m[1:]);
for name,k in [('raw',x),('cal',z),('forward',aff(x)@m),('postB',aff(z)@B),('rawB',aff(x)@B)]:
 P=quad(k)@C
 for nm,Q in [('linear',aff(k)),('quad',quad(k)),('C1',aff(P)),('C1+linear',np.c_[aff(k),P]),('C1quad',quad(P))]:
  e=Q[56:]@ls(Q[:56],W[:56])-W[56:]; print(name,nm,np.mean(abs(e),0))
