import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x): return np.c_[np.ones(len(x)),x]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
a=S['panel_a']; u=a['u']; v=a['v']; keep=np.ones(len(u),bool)
for _ in range(5):
 b=np.array([fit(aff(u[keep,j:j+1]),v[keep,j]) for j in range(4)]); r=np.linalg.norm(v-b[:,0]-u*b[:,1],axis=1); keep=r<max(.001,3*np.median(r))
z=(S['train']['x']-b[:,0])/b[:,1]; u=S['panel_b']['u']; B=fit(aff(u),S['panel_b']['v']); p=S['panel_c']; U=p['u']; Y=np.c_[S['train']['y'],S['train']['traj']]; ids=np.arange(len(U)); cv=ids%4!=0
pairs=list(combinations(range(4),2)); quad=list(combinations_with_replacement(range(4),2)); cubic=list(combinations_with_replacement(range(4),3));
def prod(x,terms): return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
F={'sin':(np.sin,1.5),'cos':(np.cos,2.),'tanh':(np.tanh,1.),'soft':(lambda x:x/(1+abs(x)),.8),'gauss':(lambda x:np.exp(-x*x),.8),'rational':(lambda x:1/(1+x*x),1.),'atan':(np.arctan,1.),'sin2':(lambda x:np.sin(x*x),1.)}
models={}
for key in ('v1','v2'):
 V=p[key]; scores=[]
 for name,g in [('z',lambda x:x),('Bz',lambda x:aff(x)@B)]:
  for fam,base in [('aff',aff),('square',lambda x:np.c_[aff(x),x*x]),('cross',lambda x:np.c_[aff(x),prod(x,pairs)]),('quad',lambda x:np.c_[aff(x),prod(x,quad)]),('addcubic',lambda x:np.c_[aff(x),x*x,x**3])]:
   G=base(g(U)); c=fit(G[cv],V[cv]); E=G@c-V; scores.append((np.mean(abs(E[cv])),np.mean(abs(E[~cv])),name,fam,None,base,g))
  for fam,(fn,start) in F.items():
   for lin in (False,True):
    base=lambda x,f,fn=fn,lin=lin:np.c_[aff(x) if lin else np.ones(len(x)),fn(x*f)]
    def r(f):
     G=base(g(U),f); return (G[cv]@fit(G[cv],V[cv])-V[cv]).ravel()
    out=min((least_squares(r,np.full(4,s),bounds=(.08,7),max_nfev=150) for s in (start,2.6,4.5)),key=lambda o:o.cost); G=base(g(U),out.x); E=G@fit(G[cv],V[cv])-V; scores.append((np.mean(abs(E[cv])),np.mean(abs(E[~cv])),name,fam+str(lin),out.x,base,g))
 print(key,'best panel held-out:');
 for row in sorted(scores,key=lambda r:r[1])[:9]:print(row[:5]);
 for j,row in enumerate(sorted(scores,key=lambda r:r[1])[:3]):
  _,_,_,_,f,base,g=row; h=lambda x:base(g(x)) if f is None else base(g(x),f); models[key+str(j)]=h(z)@fit(h(U),V)
 print('quadratic coefficients LS:'); print(np.round(fit(np.c_[aff(U),prod(U,quad)],V),3))
W=aff(z)@B; n=52
sets={'z':aff(z),'z2':np.c_[aff(z),prod(z,quad)],'z3':np.c_[aff(z),prod(z,quad),prod(z,cubic)],'B2':np.c_[aff(W),prod(W,quad)]}
for i in range(3):
 for j in range(3):
  c,d=models['v1'+str(i)],models['v2'+str(j)]; a=np.c_[aff(z),c,d]; sets[f'branches{i}{j}']=a; sets[f'cross{i}{j}']=np.c_[a,(c[:,:,None]*d[:,None,:]).reshape(len(z),-1)]
for name,a in sets.items():
 e=np.abs(a[n:]@fit(a[:n],Y[:n])-Y[n:]); print(name,np.round([e[:,0].mean(),e[:,1:].mean()],6))
