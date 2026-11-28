import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
frozen={}; exec(open('/holdout_task/_frozen_nodes/node_01.py').read(),frozen)
def affine(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
u,v=(s['panel_a'][k] for k in ('u','v'))
a=affine(u); m=np.ones(len(u),bool)
for _ in range(6):
 r=np.linalg.norm(a@ls(a[m],v[m])-v,axis=1);m=np.argsort(r)[:len(u)-3]
coef=[]
for j in range(4):
 k=(j+1)%4; D=np.c_[np.ones(len(u)),u[:,j],u[:,k],u[:,j]*u[:,k],u[:,j]**3]; c=ls(D[m],v[m,j]);coef.append(c);print('a channel',j,c.round(6),'rms',np.sqrt(np.mean((D[m]@c-v[m,j])**2)))
c=np.array(coef); b=ls(affine(s['panel_b']['u']),s['panel_b']['v'])
def A(x):
 n=np.roll(x,-1,axis=-1);return c[:,0]+c[:,1]*x+c[:,2]*n+c[:,3]*x*n+c[:,4]*x**3
def Ai(x):
 out=[]
 for row in x:
  q=least_squares(lambda z:A(z)-row,(row-c[:,0])/c[:,1],gtol=1e-12)
  out.append(q.x)
 return np.array(out)
def B(x):return affine(x)@b
def Bi(x):return np.linalg.solve(b[1:].T,(x-b[0]).T).T
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
trans={'raw':lambda x:x,'A':A,'Ai':Ai,'B':B,'Bi':Bi,'BAi':lambda x:B(Ai(x)),'BiAi':lambda x:Bi(Ai(x)),'AiBi':lambda x:Ai(Bi(x)),'AiB':lambda x:Ai(B(x)),'BA':lambda x:B(A(x)),'AB':lambda x:A(B(x))}
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; C={}
for key,fun in [('v1',np.tanh),('v2',np.cos)]:
 def basis(z,a): return np.c_[affine(z),fun(a*z)]
 def obj(a):
  D=basis(u,a); e=p[key]-D@ls(D,p[key]); return np.sum(e*e)
 a=minimize_scalar(obj,bounds=(.6,3.5),method='bounded').x
 w=ls(basis(u,a),p[key]); print(key,'frequency/RMS',a,(obj(a)/96)**.5,'coef',w.round(4)); C[key]=(lambda z,a=a,w=w,basis=basis,f=fun:np.c_[affine(z),f(a*z)]@w)
from itertools import combinations_with_replacement
for name,z in [('Ai',Ai(x)),('BiAi',Bi(Ai(x))),('AiAi',Ai(Ai(x)))]:
 print(name,'range',z.min(0),z.max(0))
 for deg in (1,2,3,4):
  D=np.column_stack([np.prod(z[:,i],axis=1) for k in range(deg+1) for i in combinations_with_replacement(range(4),k)])
  o=ls(D[:52],np.c_[y,t][:52]); e=D@o-np.c_[y,t]; print(deg, np.mean(abs(e[:52]),axis=0).round(4), np.mean(abs(e[52:]),axis=0).round(4))
V=np.linalg.svd(t-t.mean(0),full_matrices=False)[2]; print('trajectory directions',V[:3].round(3),'times',s['eval_times']); z=Ai(x); L=(t-t.mean(0))@V[:3].T; print('latent correlations',np.corrcoef(np.c_[z,Bi(z),y,L].T)[-4:,:8].round(3));
print('sorted latent rows'); print(np.c_[z,y,L][np.argsort(L[:,0])][::3].round(3));
for name,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos)]:
 for k in (.2,.5,1.,1.7,2.4,3.4):
  for a,zz in [('Ai',z),('BiAi',Bi(z))]:
   D=np.c_[affine(zz),f(k*zz)]; e=D[52:]@ls(D[:52],np.c_[y,t][:52])-np.c_[y,t][52:]; print(a,name,k,np.mean(abs(e[:,0]).round(5)),np.mean(abs(e[:,1:]).round(5)))
