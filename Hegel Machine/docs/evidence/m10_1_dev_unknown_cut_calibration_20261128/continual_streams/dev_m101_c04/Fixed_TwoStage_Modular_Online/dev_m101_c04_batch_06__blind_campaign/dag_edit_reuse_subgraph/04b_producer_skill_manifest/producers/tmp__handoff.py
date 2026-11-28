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
from itertools import combinations_with_replacement
p=s['panel_c']; u=p['u']; v1=p['v1']; v2=p['v2']
def design(x, terms):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,t],axis=1) for t in terms])
lin=[(i,) for i in range(4)];quad=list(combinations_with_replacement(range(4),2)); diag=[(i,i,i) for i in range(4)]; cyc=[(i,(i+1)%4) for i in range(4)]
for name,F in trans.items():
 for xx,yy in ((F(u),v1),(F(u),v2),(F(v1),u),(F(v2),u),(F(v1),v2)):
  D=affine(xx);e=yy-D@ls(D,yy);print(name,round(np.sqrt(np.mean(e*e)),5),end='; ')
 print()
for name,V in [('v1',v1),('v2',v2)]:
 for terms in (lin,lin+quad,lin+cyc+diag,lin+quad+diag):
  D=design(u,terms); c0=ls(D,V); e=V-D@c0; h=np.sum(np.linalg.qr(D)[0]**2,axis=1); print(name,len(terms), 'rms/loo',np.sqrt(np.mean(e**2)),np.mean(abs(e)/(1-h[:,None])))
  if len(terms)==18: print(list(zip([()]+terms,c0.round(3).tolist())))
 for f in [np.sin,np.tanh,np.cos]:
  for a0 in [.5,1.,1.7,2.,3.]:
   D=np.c_[affine(u),f(a0*u)]; e=V-D@ls(D,V); h=np.sum(np.linalg.qr(D)[0]**2,axis=1);print(name,f.__name__,a0,np.round([np.sqrt(np.mean(e**2)),np.mean(abs(e)/(1-h[:,None]))],6))
print('max abs x,u,v1,v2,Ai(x)',*[np.max(abs(q),axis=0).round(3) for q in (x,u,v1,v2,Ai(x))]); print('ranges',x.min(axis=0),x.max(axis=0))
